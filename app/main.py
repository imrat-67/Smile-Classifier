import os
from datetime import datetime

from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import engine, get_db, Base
from app.models import ClassificationHistory
from app.ml_utils import (
    convert_and_save_as_jpg,
    train_and_save_model,
    predict_image,
    clear_folder,
    save_training_info,
    load_training_info,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smile Classifier - 30227")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/saved_images", StaticFiles(directory="saved_images"), name="saved_images")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_SMILE_DIR = "uploads/smile"
UPLOAD_NON_SMILE_DIR = "uploads/non_smile"
SAVED_IMAGES_DIR = "saved_images"
MODEL_PATH = "model/smile_model.pkl"

MAX_UPLOAD_COUNT = 5000
MAX_PART_SIZE = 5000 * 1024  # 5000 KB per file, matches ml_utils.MAX_FILE_SIZE


@app.get("/")
def home(request: Request):
    """Render the home page with a short project explanation."""
    return templates.TemplateResponse(request=request, name="home.html", context={})


@app.get("/train")
def train_form(request: Request):
    """Show the training upload form plus stats from the last training run."""
    info = load_training_info()
    return templates.TemplateResponse(
        request=request, name="train.html", context={"info": info}
    )


@app.post("/train")
async def train_submit(request: Request):
    """Save uploaded images, train the model, then delete the temporary uploads."""
    try:
        form = await request.form(
            max_files=MAX_UPLOAD_COUNT,
            max_fields=MAX_UPLOAD_COUNT,
            max_part_size=MAX_PART_SIZE,
        )
        smile_files = form.getlist("smile_files")
        non_smile_files = form.getlist("non_smile_files")

        if not smile_files or not non_smile_files:
            raise ValueError("Please select images for both categories")

        os.makedirs(UPLOAD_SMILE_DIR, exist_ok=True)
        os.makedirs(UPLOAD_NON_SMILE_DIR, exist_ok=True)

        for file in smile_files:
            contents = await file.read()
            filename = os.path.splitext(file.filename)[0] + ".jpg"
            save_path = os.path.join(UPLOAD_SMILE_DIR, filename)
            convert_and_save_as_jpg(contents, save_path)

        for file in non_smile_files:
            contents = await file.read()
            filename = os.path.splitext(file.filename)[0] + ".jpg"
            save_path = os.path.join(UPLOAD_NON_SMILE_DIR, filename)
            convert_and_save_as_jpg(contents, save_path)

        accuracy, total_images = train_and_save_model(
            UPLOAD_SMILE_DIR, UPLOAD_NON_SMILE_DIR, MODEL_PATH
        )
        save_training_info(accuracy, total_images)

        clear_folder(UPLOAD_SMILE_DIR)
        clear_folder(UPLOAD_NON_SMILE_DIR)

        message = f"Model trained on {total_images} images. Test accuracy: {accuracy * 100:.2f}%"
        return templates.TemplateResponse(
            request=request,
            name="train.html",
            context={"message": message, "info": load_training_info()},
        )

    except ValueError as e:
        return templates.TemplateResponse(
            request=request,
            name="train.html",
            context={"error": str(e), "info": load_training_info()},
        )
    except Exception:
        return templates.TemplateResponse(
            request=request,
            name="train.html",
            context={"error": "Training failed. Please check your images and try again.", "info": load_training_info()},
        )


@app.get("/classify")
def classify_form(request: Request):
    """Show the single-image classify upload form."""
    return templates.TemplateResponse(request=request, name="classify.html", context={})


@app.post("/classify")
async def classify_submit(request: Request, db: Session = Depends(get_db)):
    """Save the uploaded image, classify it, store the result, and show it back with the image."""
    try:
        form = await request.form(max_part_size=MAX_PART_SIZE)
        image_file = form.get("image_file")

        if image_file is None or image_file == "":
            raise ValueError("Please select an image to classify")

        if not os.path.exists(MODEL_PATH):
            raise ValueError("No trained model found. Please train the model first.")

        os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)

        contents = await image_file.read()
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}.jpg"
        save_path = os.path.join(SAVED_IMAGES_DIR, filename)
        convert_and_save_as_jpg(contents, save_path)

        prediction = predict_image(save_path, MODEL_PATH)

        record = ClassificationHistory(image_path=save_path, predicted_class=prediction)
        db.add(record)
        db.commit()

        return templates.TemplateResponse(
            request=request,
            name="classify.html",
            context={"prediction": prediction, "image_url": f"/saved_images/{filename}"},
        )

    except ValueError as e:
        return templates.TemplateResponse(
            request=request, name="classify.html", context={"error": str(e)}
        )
    except Exception:
        return templates.TemplateResponse(
            request=request,
            name="classify.html",
            context={"error": "Classification failed. Please try a different image."},
        )


@app.get("/history")
def history(request: Request, db: Session = Depends(get_db)):
    """Show a table of all past classification results, newest first."""
    records = db.query(ClassificationHistory).order_by(
        ClassificationHistory.created_at.desc()
    ).all()
    return templates.TemplateResponse(
        request=request, name="history.html", context={"records": records}
    )


@app.post("/history/clear")
def clear_history(db: Session = Depends(get_db)):
    """Delete every row from the classification history table and their image files."""
    records = db.query(ClassificationHistory).all()
    for record in records:
        if os.path.exists(record.image_path):
            os.remove(record.image_path)
    db.query(ClassificationHistory).delete()
    db.commit()
    return RedirectResponse(url="/history", status_code=303)
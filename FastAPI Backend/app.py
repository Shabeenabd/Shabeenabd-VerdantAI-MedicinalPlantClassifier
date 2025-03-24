from fastapi import FastAPI, File, UploadFile
from main import pred_and_plot_image
from database_logging import save_to_db 

# Initializing fastapi server.
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
	image=file.file.read()
	# Get model prediction.
	pred_class = pred_and_plot_image(image_bytes=image)
	# log user input image and model prediction to database.
	save_to_db(file.filename,image,pred_class[1])	
	return {"filename": file.filename, "prediction": pred_class} 

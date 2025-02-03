from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv, find_dotenv
from boto3.exceptions import Boto3Error
from gradio_client import Client, handle_file
import uvicorn
import replicate
import boto3
import os
import requests

load_dotenv(find_dotenv())

app = FastAPI(
    title="Frame Interpolation MVP",
    description="API for Frame Interpoltion using deep learning models",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
s3_client = boto3.client(
    "s3",
    region_name=os.getenv("REGION_NAME"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
hugging_face_client = Client("AashishNandakumar/video-quality-enhancement")


async def upload_to_s3(file: UploadFile, filename: str):
    file.file.seek(0)
    try:
        s3_client.upload_fileobj(
            file.file,
            os.getenv("S3_BUCKET"),
            filename,
        )
        file_url = f"https://{os.getenv('S3_BUCKET')}.s3.{os.getenv('REGION_NAME')}.amazonaws.com/{filename}"
        return file_url
    except Boto3Error as e:
        print(f"exception while uploading file to s3: {str(e)}")
        return None


@app.get("/")
def root():
    return {"message": "welcome to frame interpolation API services"}


@app.post("/frame-interpolate/m1")
async def frame_interpolation_model_1(images: List[UploadFile] = File(...)):
    try:
        if len(images) != 2:
            return {"error": "Please upload exactly two images."}

        url_frame1 = await upload_to_s3(images[0], images[0].filename)
        url_frame2 = await upload_to_s3(images[1], images[1].filename)

        if not url_frame1 or not url_frame2:
            return {"error": "Failed to upload images to S3"}

        input = {
            "frame1": url_frame1,
            "frame2": url_frame2,
            "times_to_interpolate": 7,
        }

        output = replicate.run(
            "google-research/frame-interpolation:4f88a16a13673a8b589c18866e540556170a5bcb2ccdc12de556e800e9456d3d",
            input=input,
        )

        return StreamingResponse(
            output,
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment;filename=interpolated.mp4"},
        )
    except Exception as e:
        print(f"exception while interpolating : {str(e)}")
        raise e


@app.post("/frame-interpolate/m2")
async def frame_interpolation_model_2(video: UploadFile = File(...), fps: int = 60):
    try:
        video_url = await upload_to_s3(video, video.filename)

        if not video_url:
            return {"error": "Failed to upload video to S3"}

        output = hugging_face_client.predict(
            video_path={"video": handle_file(video_url)},
            output_fps=fps,
            api_name="/predict",
        )
        # print(f"enhanced FPS output: {output}")

        # response = requests.get(f"file://{output_path}", stream=True)
        # response.raise_for_status()

        return StreamingResponse(
            open(output, "rb"),
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment;filename=interpolated.mp4"},
        )
    except Exception as e:
        print(f"exception while interpolating : {str(e)}")
        raise e


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="trace")

# =========================
#  Module: Video Conversion and Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import datetime
import csv
from transformers import pipeline
from moviepy import VideoFileClip
import os
from langchain_community.document_loaders import CSVLoader

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build(video_file):
    csv_output_path = convert_mp4_to_wav_single_arg(video_file)
    loader = CSVLoader(file_path=csv_output_path, encoding="utf-8", csv_args={'delimiter': ','})
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

def convert_mp4_to_wav_single_arg(mp4_file):
    # Output filenames
    local_mp4_file = "uploaded_video.mp4"
    wav_file = "output.wav"

    # Save the UploadedFile object to a local file
    with open(local_mp4_file, "wb") as f:
        f.write(mp4_file.read())
        print(f"Uploaded video saved locally as {local_mp4_file}")

    # Delete existing output.wav if present
    if os.path.exists(wav_file):
        os.remove(wav_file)
        print("Existing output.wav deleted.")

    # Extract audio from MP4
    video_clip = VideoFileClip(local_mp4_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(wav_file)
    audio_clip.close()
    video_clip.close()

    print(f"WAV file saved successfully: {wav_file}")

    # Transcribe audio to text
    whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium', device=0)
    transcription = whisper(wav_file, return_timestamps=True)

    # Format transcription and save to CSV
    formatted_data = format_timestamps(transcription)
    csv_output = write_to_csv(formatted_data, 'output.csv')
    print(f"CSV saved at: {csv_output}")

    # Cleanup local MP4 file
    os.remove(local_mp4_file)

    return csv_output


def format_timestamps(data):
    """Formats timestamps and extracts text into a list of dictionaries."""
    formatted_data = []
    for item in data['chunks']:
        start_time, end_time = item['timestamp']
        text = item['text']
        if end_time == 0.0:
            timestamp_str = str(datetime.timedelta(seconds=start_time))
        else:
            timestamp_str = f"{str(datetime.timedelta(seconds=start_time))} - {str(datetime.timedelta(seconds=end_time))}"
        formatted_data.append({"Timestamp": timestamp_str, "Text": text})
    return formatted_data


def write_to_csv(formatted_data, csv_file):
    """Writes the formatted data to a CSV file."""
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Timestamp", "Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_data)
    return csv_file

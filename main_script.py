import whisper
from text_processor import answer_question
model = whisper.load_model("base")


def generate_text_from_video(video_link: str) -> str:
    audio = whisper.load_audio(video_link)
    return model.transcribe(audio)["text"]


def get_answer(question: str, video_link: str) -> str:
    context = generate_text_from_video(video_link)
    answer = answer_question(context, question)
    print("The answer provided is: {}".format(answer))


if __name__ == "__main__":
    video_link = "/Users/pragathi/Downloads/harvey_specter.mp4"
    question = "How is life?"
    get_answer(question, video_link)
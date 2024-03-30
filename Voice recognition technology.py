import speech_recognition as sr
from huggingface_hub import InferenceClient
r = sr.Recognizer()
with sr.Microphone() as source:
    audio = r.listen(source)
try:
    text = r.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")
def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(prompt, history=None):
    if history is None:
        history = []
    temperature = 0.9
    max_new_tokens = 3000
    top_p = 0.95
    repetition_penalty = 1.0

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )
    prompts = format_prompt(prompt, history)
    stream = client.text_generation(
        prompts,
        stream=True,
        details=True,
        return_full_text=False,
        **generate_kwargs
    )

    output = ""

    for response in stream:
        output += response.token.text
    return output

print(generate(text))
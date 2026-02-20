import gradio as gr
from core.agent import run_pipeline


async def chat(message, history):
    return await run_pipeline(message)


if __name__ == "__main__":
    gr.ChatInterface(chat).launch()

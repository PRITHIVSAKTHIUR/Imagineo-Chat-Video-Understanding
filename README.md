---
title: IMAGINEO CHAT
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: compose chat multi-modal
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Imagineo Chat

Imagineo Chat is a versatile multimodal chatbot that integrates text, image, and video processing capabilities. It leverages state-of-the-art models like **Gemma3-4B**, **Qwen2-VL-OCR**, and **Stable Diffusion XL** to provide a seamless conversational experience. Whether you want to generate images, analyze videos, or have a text-based conversation, Imagineo Chat has you covered.

## Features

- **Text Generation**: Engage in natural language conversations with the chatbot.
- **Image Generation**: Generate high-quality images using Stable Diffusion XL models (Lightning 5, Lightning 4, Turbo v3).
- **Multimodal Processing**: Analyze and describe images using the **Gemma3-4B** model.
- **Video Analysis**: Process and summarize video content using the **Gemma3-4B** model.
- **Text-to-Speech (TTS)**: Convert text responses into speech using **Edge TTS**.
- **OCR (Optical Character Recognition)**: Extract and interpret text from images using **Qwen2-VL-OCR**.

## How to Use

### Text and Chat
Simply type your query in the text box. You can also use special tags to trigger specific functionalities:

- **@gemma3-4b**: Use this tag for multimodal queries involving images or text.
- **@gemma3-4b-video**: Use this tag to analyze and describe video content.
- **@lightningv5**: Generate images using the **Stable Diffusion XL Lightning 5** model.
- **@lightningv4**: Generate images using the **Stable Diffusion XL Lightning 4** model.
- **@turbov3**: Generate images using the **Stable Diffusion XL Turbo v3** model.
- **@tts1** or **@tts2**: Convert the chatbot's response into speech using different voices.

### Image and Video Upload
You can upload images or videos directly into the chat interface. The chatbot will process the media and provide relevant responses based on the query.

### Examples
Here are some example queries you can try:

- **@gemma3-4b Explain the Image** (Upload an image)
- **@gemma3-4b-video Explain what is happening in this video?** (Upload a video)
- **@lightningv5 Chocolate dripping from a donut**
- **@tts1 Who is Nikola Tesla, and why did he die?**
- **@lightningv4 Cat holding a sign that says hello world**
- **@turbov3 Anime illustration of a wiener schnitzel**
- **@tts2 What causes rainbows to form?**

## Installation

To run Imagineo Chat locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/imagineo-chat.git
   cd imagineo-chat
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Chat Interface**:
   Open your browser and navigate to `http://localhost:7860` to access the chat interface.

## Models Used

- **Text Generation**: `prithivMLmods/FastThink-0.5B-Tiny`
- **Multimodal (OCR)**: `prithivMLmods/Qwen2-VL-OCR-2B-Instruct`
- **Image Generation**:
  - **Lightning 5**: `SG161222/RealVisXL_V5.0_Lightning`
  - **Lightning 4**: `SG161222/RealVisXL_V4.0_Lightning`
  - **Turbo v3**: `SG161222/RealVisXL_V3.0_Turbo`
- **Multimodal (Text & Image)**: `google/gemma-3-4b-it`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Gradio
- Edge TTS
- OpenCV (for video processing)
- Diffusers (for image generation)

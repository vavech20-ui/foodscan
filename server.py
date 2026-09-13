from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import json
import os
import base64
import io

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """Ты — эксперт по анализу состава продуктов питания.
Проанализируй фото этикетки и верни СТРОГО валидный JSON без markdown и пояснений:

{
  "safety_rating": "safe" | "warning" | "danger",
  "ai_analysis": "Краткий анализ состава на русском (2-4 предложения)",
  "extracted_text": "Полный текст с этикетки, если удалось прочитать",
  "additives": [
    {
      "name": "Название компонента или E-код",
      "description": "Влияние на здоровье (1-2 предложения)",
      "danger_level": "safe" | "warning" | "danger"
    }
  ]
}

Критерии:
- safe: натуральные ингредиенты, безопасные добавки
- warning: спорные добавки, сахар, избыток соли, пальмовое масло
- danger: канцерогены (E123, E250, нитриты), транс-жиры, запрещённые вещества

Если состав не виден — верни safety_rating: "warning" и объясни в ai_analysis.
В additives добавляй только компоненты, вызывающие вопросы."""


def compress_image(image_base64: str) -> str:
    try:
        from PIL import Image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        img_bytes = base64.b64decode(image_base64)
        if len(img_bytes) <= 1_500_000:
            return image_base64
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        max_side = 1600
        ratio = min(max_side / img.width, max_side / img.height, 1.0)
        if ratio < 1.0:
            img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=82, optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f'Compression error: {e}')
        return image_base64.split(',')[1] if ',' in image_base64 else image_base64


def extract_json(text: str) -> dict:
    text = text.strip()
    if '```json' in text:
        text = text.split('```json', 1)[1].split('```', 1)[0]
    elif '```' in text:
        text = text.split('```', 1)[1].split('```', 1)[0]
    return json.loads(text)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        image = data.get('image') if data else None
        if not image:
            return jsonify({'error': 'Изображение не получено'}), 400

        image_b64 = compress_image(image)

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return jsonify({'error': 'GEMINI_API_KEY не задан'}), 500

        client = OpenAI(
            api_key=api_key,
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
        resp = client.chat.completions.create(
            model='gemini-2.0-flash',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                    {'type': 'text', 'text': 'Проанализируй состав продукта на фото.'}
                ]}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        result = extract_json(resp.choices[0].message.content)
        result.setdefault('safety_rating', 'warning')
        result.setdefault('additives', [])
        result.setdefault('ai_analysis', '')
        result.setdefault('extracted_text', '')
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({'error': f'Модель вернула не-JSON: {str(e)[:100]}'}), 500
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': f'Ошибка: {str(e)[:200]}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
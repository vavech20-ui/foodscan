from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
import base64

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """Ты — помощник по анализу состава продуктов питания.
Проанализируй фото этикетки и верни СТРОГО валидный JSON без markdown.

ВАЖНО: в additives включай ТОЛЬКО самые важные компоненты (максимум 8-10 штук).
Не включай обычные ингредиенты: воду, муку, сахар, соль, молоко, яйца.
Фокусируйся на добавках (E-коды), консервантах, красителях, усилителях вкуса.

Формат:
{
  "safety_rating": "safe" | "warning" | "danger",
  "ai_analysis": "Краткий анализ на русском (2-3 предложения)",
  "extracted_text": "Ключевые компоненты с этикетки (до 300 символов)",
  "additives": [
    {
      "name": "Название или E-код",
      "description": "1 предложение о влиянии",
      "danger_level": "safe" | "warning" | "danger"
    }
  ]
}

Критерии:
- safe: натуральные ингредиенты
- warning: сахар, пальмовое масло, искусственные красители
- danger: не рекомендуемые E-добавки, нитриты

Если состав не виден — safety_rating: "warning" и объясни в ai_analysis."""


def decode_image(image_base64: str) -> str:
    if ',' in image_base64:
        return image_base64.split(',')[1]
    return image_base64


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

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return jsonify({'error': 'GEMINI_API_KEY не задан'}), 500

        image_b64 = decode_image(image)

        # Используем gemini-3.6-flash как просил пользователь
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}'

        payload = {
            "contents": [{
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\nПроанализируй состав продукта на фото. Кратко, только важные компоненты."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800
            }
        }

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 429:
            return jsonify({'error': 'Превышен лимит запросов. Подождите минуту.'}), 429

        if response.status_code == 503:
            return jsonify({'error': 'Сервис временно недоступен. Попробуйте через минуту.'}), 503

        if response.status_code != 200:
            error_text = response.text[:200]
            return jsonify({'error': f'Ошибка API ({response.status_code}): {error_text}'}), 500

        resp_json = response.json()

        try:
            text = resp_json['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            block_reason = resp_json.get('promptFeedback', {}).get('blockReasonMessage', 'Неизвестно')
            return jsonify({
                'error': f'Не удалось проанализировать. Попробуйте другое фото. ({block_reason})'
            }), 400

        try:
            result = extract_json(text)
        except json.JSONDecodeError:
            return jsonify({
                'safety_rating': 'warning',
                'ai_analysis': 'Не удалось разобрать состав. Попробуйте фото при лучшем свете.',
                'extracted_text': text[:300] if text else '',
                'additives': []
            })

        result.setdefault('safety_rating', 'warning')
        result.setdefault('additives', [])
        result.setdefault('ai_analysis', '')
        result.setdefault('extracted_text', '')

        return jsonify(result)

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': f'Ошибка сервера: {str(e)[:150]}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
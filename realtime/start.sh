export OPENAI_API_KEY=your-api-key-here
export OPENAI_MODEL=granite-4.0-h-350m
export OPENAI_TTS_VOICE=af_heart
export OPENAI_TTS_MODEL=kokoro
export OPENAI_TTS_BASE_URL=http://192.168.1.111:8884/v1
export OPENAI_BASE_URL=http://192.168.1.111:8880/v1
export WHISPER_MODEL=whisper-1
export WHISPER_LANGUAGE=en
export SYSTEM_PROMPT="You are a helpful and friendly assistant. Provide clear, accurate, and concise responses."

bash run.sh

# Environment Setup

This document explains how to set up the required environment variables for the Stock Tracker application, both for local development and production deployment.

## API Credentials

The Stock Tracker application uses AI-powered features that require API credentials to function. These credentials should be kept secure and never committed to version control.

## Local Development Setup

For local development, create a `.env` file in the root of the project with the following content:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_API_URL=https://free.v36.cm/v1/chat/completions
```

Replace `your_api_key_here` with your actual API key.

> **Note**: The `.env` file is included in `.gitignore` to prevent accidentally committing it to version control.

You can use the `env.example` file as a template:

```bash
cp env.example .env
```

Then edit the `.env` file to add your actual API credentials.

## Production Deployment

In a production environment, you should set environment variables directly on the server or through your hosting provider. Do not use a `.env` file in production unless it's properly secured.

### Setting Environment Variables in Different Environments

#### GitHub Actions

```yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      OPENAI_API_URL: ${{ secrets.OPENAI_API_URL }}
```

#### Heroku

```bash
heroku config:set OPENAI_API_KEY=your_api_key_here
heroku config:set OPENAI_API_URL=https://free.v36.cm/v1/chat/completions
```

#### Docker

```bash
docker run -e OPENAI_API_KEY=your_api_key_here -e OPENAI_API_URL=https://free.v36.cm/v1/chat/completions stock-tracker
```

## Available Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your API key | (required) |
| `OPENAI_API_URL` | API endpoint URL | https://free.v36.cm/v1/chat/completions |
| `OPENAI_MODEL` | AI model to use | gpt-4o-mini |
| `OPENAI_MAX_TOKENS` | Maximum tokens for API responses | 1000 |
| `OPENAI_TEMPERATURE` | Randomness of API responses (0.0-1.0) | 0.7 |
| `OPENAI_REQUEST_TIMEOUT` | API request timeout in seconds | 30 |
| `CACHE_ENABLED` | Enable API response caching | True |
| `CACHE_EXPIRY` | Cache expiry time in seconds | 3600 |
| `MAX_RETRIES` | Maximum API request retries | 3 |

## Security Best Practices

1. Never commit API keys or secrets to version control
2. Rotate API keys regularly
3. Use environment-specific settings (development vs. production)
4. Implement proper access controls for API keys
5. Consider using a secrets management solution for production deployments 
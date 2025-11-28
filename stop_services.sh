#!/bin/bash
# Stop script for YTSEED pipeline services

echo "🛑 Stopping YTSEED Pipeline Services..."
echo ""

# Stop FastAPI
echo "🌐 Stopping FastAPI..."
pkill -f "uvicorn main:app"
if [ $? -eq 0 ]; then
    echo "   ✅ FastAPI stopped"
else
    echo "   ℹ️  FastAPI not running"
fi

# Stop Celery
echo ""
echo "⚙️  Stopping Celery..."
pkill -f "celery.*worker"
if [ $? -eq 0 ]; then
    echo "   ✅ Celery stopped"
else
    echo "   ℹ️  Celery not running"
fi

# Stop Redis (optional - usually keep running)
echo ""
echo "📦 Redis Status:"
if pgrep -x "redis-server" > /dev/null; then
    echo "   ℹ️  Redis still running (intentional - shared service)"
    echo "   To stop Redis manually: redis-cli shutdown"
else
    echo "   ✅ Redis not running"
fi

echo ""
echo "✅ Services stopped!"

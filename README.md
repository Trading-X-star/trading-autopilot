# 🤖 Trading Autopilot

Автоматическая торговая система для MOEX с поддержкой до 3 аккаунтов.

## 🚀 Быстрый старт

```bash
# Клонируем и запускаем
cd trading-autopilot
chmod +x scripts/start.sh
./scripts/start.sh
```

## 📊 Сервисы

| Сервис | Порт | Описание |
|--------|------|----------|
| **Dashboard** | 8022 | Web-интерфейс |
| **Account Manager** | 8020 | Управление аккаунтами (до 3) |
| **Data Ingestion** | 8002 | Данные MOEX |
| **Strategy** | 8005 | Торговые сигналы |
| **Execution** | 8003 | Исполнение ордеров |
| **Risk Manager** | 8001 | Риск-менеджмент |
| **Trailing Stop** | 8023 | Динамические стоп-лоссы |
| **Profit Distribution** | 8024 | Распределение прибыли |
| **Grafana** | 3000 | Мониторинг (admin/admin123) |
| **Prometheus** | 9090 | Метрики |

## 📝 API Примеры

### Создать аккаунт
```bash
curl -X POST http://localhost:8020/accounts \
  -H "Content-Type: application/json" \
  -d '{"name": "Main", "risk_profile": "balanced", "initial_balance": 1000000}'
```

### Профили риска
- `conservative` — макс. позиция 5%, стоп 3%
- `balanced` — макс. позиция 10%, стоп 5%
- `aggressive` — макс. позиция 20%, стоп 7%

### Получить котировку
```bash
curl http://localhost:8002/price/SBER
```

### Анализ тикера
```bash
curl http://localhost:8005/analyze/SBER
```

### Список аккаунтов
```bash
curl http://localhost:8020/accounts
```

### Позиции аккаунта
```bash
curl http://localhost:8020/accounts/{id}/positions
```

## 🔧 Команды

```bash
# Логи
docker-compose logs -f

# Статус сервисов
docker-compose ps

# Остановить
docker-compose down

# Перезапустить сервис
docker-compose restart account-manager

# Пересобрать
docker-compose up -d --build
```

## 📁 Структура

```
trading-autopilot/
├── docker-compose.yml
├── services/
│   ├── orchestrator/
│   ├── data-ingestion/
│   ├── risk-manager/
│   ├── execution/
│   ├── portfolio/
│   ├── strategy/
│   ├── account-manager/
│   ├── trailing-stop/
│   ├── profit-distribution/
│   ├── decision-router/
│   ├── multi-dashboard/
│   ├── alert-manager/
│   └── health-monitor/
├── monitoring/
├── configs/
└── scripts/
```

## ⚙️ Настройка Telegram

1. Создайте бота через @BotFather
2. Получите CHAT_ID через @userinfobot
3. Добавьте в `.env`:
```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789
```

## 📈 Функции

- ✅ До 3 торговых аккаунтов
- ✅ Trailing stop-loss с автоматическим breakeven
- ✅ Автораспределение 10% прибыли на основной счёт
- ✅ Риск-менеджмент по профилям
- ✅ Real-time данные MOEX
- ✅ Telegram уведомления
- ✅ Web Dashboard
- ✅ Prometheus + Grafana мониторинг

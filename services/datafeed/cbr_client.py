import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import xml.etree.ElementTree as ET

class CBRClient:
    """Клиент для API Центрального Банка России"""
    
    BASE_URL = "https://www.cbr.ru"
    JSON_DAILY = "https://www.cbr-xml-daily.ru/daily_json.js"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    async def get_currency_rates(self) -> dict:
        """Текущие курсы валют"""
        async with self.session.get(self.JSON_DAILY) as resp:
            data = await resp.json(content_type=None)
            return {
                'date': data['Date'],
                'usd': data['Valute']['USD']['Value'],
                'usd_prev': data['Valute']['USD']['Previous'],
                'eur': data['Valute']['EUR']['Value'],
                'eur_prev': data['Valute']['EUR']['Previous'],
                'cny': data['Valute']['CNY']['Value'],
                'usd_change': data['Valute']['USD']['Value'] - data['Valute']['USD']['Previous'],
                'eur_change': data['Valute']['EUR']['Value'] - data['Valute']['EUR']['Previous'],
            }
    
    async def get_key_rate(self) -> dict:
        """Ключевая ставка ЦБ"""
        url = f"{self.BASE_URL}/scripts/XML_key_rate.asp"
        async with self.session.get(url) as resp:
            text = await resp.text()
            root = ET.fromstring(text)
            
            rates = []
            for record in root.findall('.//Record'):
                rates.append({
                    'date': record.get('Date'),
                    'rate': float(record.find('Rate').text.replace(',', '.'))
                })
            
            # Последняя ставка
            if rates:
                latest = max(rates, key=lambda x: datetime.strptime(x['date'], '%d.%m.%Y'))
                return {
                    'current_rate': latest['rate'],
                    'date': latest['date'],
                    'history': rates[-10:]  # последние 10 изменений
                }
            return {}
    
    async def get_key_rate_history(self, days: int = 365) -> pd.DataFrame:
        """История ключевой ставки"""
        url = f"{self.BASE_URL}/scripts/XML_key_rate.asp"
        async with self.session.get(url) as resp:
            text = await resp.text()
            root = ET.fromstring(text)
            
            records = []
            for record in root.findall('.//Record'):
                records.append({
                    'date': datetime.strptime(record.get('Date'), '%d.%m.%Y'),
                    'key_rate': float(record.find('Rate').text.replace(',', '.'))
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('date')
            return df
    
    async def get_currency_history(self, currency: str = 'USD', 
                                   days: int = 365) -> pd.DataFrame:
        """История курса валюты"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Код валюты для USD = R01235
        currency_codes = {'USD': 'R01235', 'EUR': 'R01239', 'CNY': 'R01375'}
        code = currency_codes.get(currency, 'R01235')
        
        url = (f"{self.BASE_URL}/scripts/XML_dynamic.asp?"
               f"date_req1={start_date.strftime('%d/%m/%Y')}&"
               f"date_req2={end_date.strftime('%d/%m/%Y')}&"
               f"VAL_NM_RQ={code}")
        
        async with self.session.get(url) as resp:
            text = await resp.text()
            root = ET.fromstring(text)
            
            records = []
            for record in root.findall('.//Record'):
                records.append({
                    'date': datetime.strptime(record.get('Date'), '%d.%m.%Y'),
                    f'{currency.lower()}_rate': float(record.find('Value').text.replace(',', '.'))
                })
            
            return pd.DataFrame(records).sort_values('date')


# Пример использования
async def main():
    async with CBRClient() as cbr:
        # Текущие курсы
        rates = await cbr.get_currency_rates()
        print(f"USD: {rates['usd']:.2f} ({rates['usd_change']:+.2f})")
        print(f"EUR: {rates['eur']:.2f}")
        
        # Ключевая ставка
        key_rate = await cbr.get_key_rate()
        print(f"Ключевая ставка: {key_rate['current_rate']}%")
        
        # История для фичей
        usd_history = await cbr.get_currency_history('USD', days=30)
        print(usd_history.tail())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

SELECT Category, COUNT(*) AS Devices, ROUND(AVG(Price),2) AS AvgPrice, ROUND(AVG(RAM_GB),2) AS AvgRAM
FROM laptops GROUP BY Category ORDER BY AvgPrice DESC;

WITH brand_prices AS (
 SELECT Manufacturer, COUNT(*) AS Devices, AVG(Price) AS AvgPrice
 FROM laptops GROUP BY Manufacturer
)
SELECT Manufacturer, Devices, ROUND(AvgPrice,2) AS AvgPrice
FROM brand_prices WHERE Devices >= 5 ORDER BY AvgPrice DESC;

SELECT Manufacturer, Category, "Model Name", Price,
 RANK() OVER (PARTITION BY Category ORDER BY Price DESC) AS PriceRankInCategory
FROM laptops ORDER BY Category, PriceRankInCategory LIMIT 40;

# YANDEX ALGORITHMS 3.0 FINAL PART

## :white_check_mark:A.[Подземная доставка](https://contest.yandex.ru/contest/46304/problems/A/)

Для ускорения работы служб доставки под городом Длинноградом был прорыт тоннель, по которому ходит товарный поезд, останавливающийся на 
промежуточных станциях возле логистических центров. На станциях к концу поезда могут быть присоединены вагоны с определенными товарами, а также 
от его конца может быть отцеплено некоторое количество вагонов или может быть проведена ревизия, во время которой подсчитывается количество 
вагонов с определенным товаром.
Обработайте операции в том порядке, в котором они производились, и ответьте на запросы ревизии.

```
from collections import deque
from collections import defaultdict


class T:
    def __init__(self) -> None:
        self.prod_register = defaultdict(int)
        self.carriages = deque(maxlen=10 ** 9)
    
    def add(self, carriage_count: int, product: str):
        if isinstance(carriage_count, str):
            carriage_count = int(carriage_count)
        self.prod_register[product] += carriage_count
        self.carriages.append((product, carriage_count))

    def delete(self, delete_carr_count: int):
        if isinstance(delete_carr_count, str):
            delete_carr_count = int(delete_carr_count)
        while delete_carr_count > 0:
            product, prod_count = self.carriages.pop()
            if delete_carr_count >= prod_count:
                self.prod_register[product] -= prod_count
                delete_carr_count -= prod_count
            else:
                prod_count -= delete_carr_count
                self.prod_register[product] -= delete_carr_count
                delete_carr_count = 0
                self.carriages.append((product, prod_count))
                break

    def get(self, product: str):
        print(self.prod_register.get(product, 0))

    def exec(self, command_line: str):
        if command_line.strip():
            command, *args = command_line.split()
            getattr(self, command)(*args)


t = T()
with open('input.txt') as file:
    n = int(next(file))
    for _ in range(n):
        cmd = next(file)        
        t.exec(cmd)
```

import numpy as np
def game_core_v3(number: int = 1) -> int:
    """Сначала устанавливаем предположительное число равное 0, а потом , используя бинарный поиск, уменьшаем
    или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    count = 0
    min_val=1
    max_val=100
    predict=0

    while number != predict:
        count += 1
        predict=(min_val+max_val)//2
        if predict < number:
            min_val = predict + 1
        else:
            max_val=predict - 1

    return count
print(f'Количество попыток: {game_core_v3()}')


def score_game(game_core_v3) -> int:
    """За какое количество попыток в среднем за 10000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    #np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(10000))  # загадали список чисел

    for number in random_array:
        count_ls.append(game_core_v3(number))
    score = int(np.mean(count_ls)) # почему то score не виден 

print(f"Ваш алгоритм угадывает число в среднем за: {score} попытки")
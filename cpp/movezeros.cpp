#include <iostream>
#include <algorithm>
#include <vector>
/*
Условие: Дан массив чисел nums. Напиши функцию, которая переместит все 0 в конец массива, сохраняя порядок остальных элементов. Важно: Сделай это in-place (без создания копии массива).
Пример: Input: [0,1,0,3,12] -> Output: [1,3,12,0,0]
*/


template<typename T>
void MoveZeros(std::vector<T>& a) {
  for (int i = 0; i < a.size() - 1; i++) {
    if (a[i] == 0) {
      for (int j = i + 1; j < a.size(); j++){
        if (a[j] != 0) {
          std::swap(a[i], a[j]);
          break;
        }
      }
    }
  }
}

int main() {
  std::vector<int> a = {0, 2, 0, 0, 1, 12};
  MoveZeros(a);
  for (int x : a) {
    std::cout << x << ' ';
  }
  return 0;
}
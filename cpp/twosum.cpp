#include <iostream>
#include <map>
#include <vector>
/*
Условие: Дан массив чисел nums и целое число target. Верни индексы двух чисел так, чтобы их сумма была равна target.

Пример: nums = [2,7,11,15], target = 9 -> Output: [0, 1] (так как 2 + 7 = 9). 
*/

const std::array<int, 2> ToSum(std::vector<int>& nums, int target) {
  std::unordered_map<int,int> s = {};
  for (int i = 0 ; i < nums.size(); i++) {
    if (s.find(target - nums[i]) != s.end()){
      return {s[target - nums[i]], i};
    } else {
      s[nums[i]] = i;
    }
  }
  return {-1, -1};
}

int main() {
  std::vector<int> a = {-1, -2, -3, -4, -5};
  int target = 10;
  std::array<int,2> answ = ToSum(a, target);
  for (int x : answ){
    std::cout << x << " ";
  }

}
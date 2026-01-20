int main() {
  std::vector<int> a = {0, 1, 0, 0, 1, 12};
  MoveZeros(a);
  for (int x : a) {
    std::cout << x << ' ';
  }
  return 0;
}
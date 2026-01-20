#include <iostream>
#include <string>


    bool compare(std::string a) {
      bool flag = 1;
      for (int i = 0; i < a.size()/2; i++) {
        std::cout << a[i] << a[a.size() - 1 - i] << std::endl;
        if (a[i] != a[a.size() - 1 - i]) {            
          flag = 0;
          break;
        }
      }
      return flag;
    }
    bool isPalindrome(std::string s) {
        std::string news = "";
        for (char i : s) {
            if ((i<92)&&(i>64)) {
              i+=32;
              ///std::cout << "if1 " << i << std::endl;
            } 
            if ((i >= 97)&&(i <= 122)) {
              news += i;
              ///std::cout << "if2 " << news << std::endl;
            } 
        }
        std::cout << news;
        return compare(news);
    }


int main() {
  std::string s = "A man, a plan, a canal: Panama";
  std::cout << isPalindrome(s);
}
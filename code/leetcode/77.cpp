#include <iostream>
#include <vector>

/*
给定一个数字n和一个整数k，输出1~n中选取k个数字的全排列
*/
void backtracking(int n, int k, int visited, std::vector<int> &path,
                  std::vector<std::vector<int>> &ans) {
  if (path.size() == k) {
    ans.push_back(path);
    return;
  }

  for (int i = visited; i <= n; i++) {
    path.push_back(i);
    backtracking(n, k, i + 1, path, ans); // 递归子节点是i+1
    path.pop_back();
  }
}

std::vector<std::vector<int>> combine(int n, int k) {
  std::vector<int> path;
  std::vector<std::vector<int>> ans;
  backtracking(n, k, 1, path, ans);
  return ans;
}

int main() {
  int n = 4, k = 2;
  auto ans = combine(n, k);
  for (int i = 0; i < ans.size(); i++) {
    std::cout << "[";
    for (int j = 0; j < ans[i].size(); j++) {
      std::cout << ans[i][j] << ",";
    }
    std::cout << "]" << std::endl;
  }
  return 0;
}
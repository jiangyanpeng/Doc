#include <iostream>
#include <stack>
#include <vector>

/*
给定一个二维0-1矩阵，表示朋友之间的关系，(i,j)=1，表示i和j之间是朋友，输出朋友圈的个数
*/

/*
int findCircleNum(std::vector<std::vector<int>> &friends) {
  int ans = 0;
  int n = friends.size();
  std::vector<bool> visited(n, false);
  for (int i = 0; i < n; i++) {
    if (!visited[i]) {
      ans++;
      visited[i] = true;
      std::stack<int> s;
      s.push(i);
      while (!s.empty()) {
        auto pos = s.top();
        s.pop();
        for (int k = 0; k < n; k++) {
          if (friends[pos][k] != 1 || visited[k])
            continue;
          s.push(k);
          visited[k] = true;
        }
      }
    }
  }
  return ans;
}
*/

void dfs(std::vector<std::vector<int>> &friends, int pos,
         std::vector<bool> &visited) {
  visited[pos] = true;
  for (int i = 0; i < friends.size(); i++) {
    if (friends[pos][i] == 1 && !visited[i]) {
      dfs(friends, i, visited);
    }
  }
}

int findCircleNum(std::vector<std::vector<int>> &friends) {
  int n = friends.size();
  int ans = 0;
  std::vector<bool> visited(n, false);
  for (int i = 0; i < n; i++) {
    if (!visited[i]) {
      ans++;
      dfs(friends, i, visited);
    }
  }
  return ans;
}
int main() {
  std::vector<std::vector<int>> friends = {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}};
  std::cout << "findCircleNum: " << findCircleNum(friends) << std::endl;
  return 0;
}
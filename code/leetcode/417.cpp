#include <iostream>
#include <vector>

/*
一个非负的二维矩阵，数值表示海拔高度，四周都是海洋，水从高处往低处流动，
输出能流到海洋的坐标数组
*/

void dfs(std::vector<std::vector<int>> &matrix, int x, int y,
         std::vector<std::vector<bool>> &visited) {
  if (visited[x][y])
    return;
  visited[x][y] = true;
  int dx[4] = {0, -1, 0, 1}, dy[4] = {1, 0, -1, 0};
  for (int i = 0; i < 4; i++) {
    int new_x = x + dx[i], new_y = y + dy[i];
    if (new_x < 0 || new_x >= matrix.size() || new_y < 0 ||
        new_y >= matrix.at(0).size() || visited[new_x][new_y] ||
        matrix[x][y] > matrix[new_x][new_y])
      continue;
    dfs(matrix, x + dx[i], y + dy[i], visited);
  }
}

std::vector<std::vector<int>>
pacificAtlantic(std::vector<std::vector<int>> &matrix) {
  int n = matrix.size(), m = n ? matrix.at(0).size() : 0;
  std::vector<std::vector<bool>> can_visited_1(n, std::vector<bool>(m, false));
  std::vector<std::vector<bool>> can_visited_2(n, std::vector<bool>(m, false));
  for (int i = 0; i < n; i++) {
    dfs(matrix, i, 0, can_visited_1);
    dfs(matrix, i, m - 1, can_visited_2);
  }

  for (int i = 0; i < m; i++) {
    dfs(matrix, 0, i, can_visited_1);
    dfs(matrix, n - 1, i, can_visited_2);
  }

  std::vector<std::vector<int>> ans;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (can_visited_1[i][j] && can_visited_2[i][j] == can_visited_1[i][j]) {
        ans.push_back({i, j});
      }
    }
  }
  return ans;
}

int main() {
  std::vector<std::vector<int>> matrix = {{1, 2, 2, 3, 5},
                                          {3, 2, 3, 4, 4},
                                          {2, 4, 5, 3, 1},
                                          {6, 7, 1, 4, 5},
                                          {5, 1, 1, 2, 4}};

  auto ans = pacificAtlantic(matrix);
  for (int i = 0; i < ans.size(); i++) {
    std::cout << "[" << ans[i][0] << "," << ans[i][1] << "]" << std::endl;
  }
  return 0;
}
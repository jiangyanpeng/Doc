#include <algorithm>
#include <iostream>
#include <stack>
#include <vector>

/*
给定一个二维矩阵，其中0表示海洋，1表示陆地，单独的陆地或者联通区域会形成岛屿，输出岛屿的数量
*/

/*
// 解法一
int maxAreaOfIsland(std::vector<std::vector<int>> &grid) {
  int n = grid.size(), m = n ? grid[0].size() : 0;
  int ans = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      int area = 0;
      if (grid[i][j] == 1) {
        grid[i][j] = 0;
        area = 1;

        std::stack<std::pair<int, int>> s;
        s.push({i, j});
        while (!s.empty()) {
          auto pos = s.top();
          s.pop();

          int dx[4] = {0, -1, 0, 1}, dy[4] = {1, 0, -1, 0};
          for (int k = 0; k < 4; k++) {
            int x = pos.first + dx[k], y = pos.second + dy[k];
            if (x < 0 || x >= n || y < 0 || y >= m || grid[x][y] != 1)
              continue;
            grid[x][y] = 0;
            s.push({x, y});
            area++;
          }
        }
      }
      ans = std::max(ans, area);
    }
  }
  return ans;
}
*/

// 接法二
int dfs(std::vector<std::vector<int>> &grid, int x, int y) {
  if (x < 0 || x >= grid.size() || y < 0 || y >= grid.at(0).size() ||
      grid[x][y] != 1)
    return 0;

  int area = 1;
  grid[x][y] = 0;

  int dx[4] = {0, -1, 0, 1}, dy[4] = {1, 0, -1, 0};
  for (int k = 0; k < 4; k++) {
    area += dfs(grid, x + dx[k], y + dy[k]);
  }
  return area;
}

int maxAreaOfIsland(std::vector<std::vector<int>> &grid) {
  int n = grid.size(), m = n ? grid.at(0).size() : 0;
  int ans = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (grid[i][j]) {
        ans = std::max(ans, dfs(grid, i, j));
      }
    }
  }
  return ans;
}

int main() {
  std::vector<std::vector<int>> grid = {{1, 0, 1, 1, 0, 1, 0, 1},
                                        {1, 0, 1, 1, 0, 1, 1, 1},
                                        {0, 0, 0, 0, 0, 0, 0, 1}};
  std::cout << "maxAreaOfIsland: " << maxAreaOfIsland(grid) << std::endl;
  return 0;
}
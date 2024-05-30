#include <iostream>
#include <stack>
#include <string>
#include <vector>

bool dfs(std::vector<std::vector<char>> &board, int x, int y, std::string &word,
         int pos) {
  if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size() ||
      word[pos] != board[x][y])
    return false;

  if (pos == word.size() - 1)
    return true;
  char tmp = board[x][y];
  board[x][y] = '*';
  int dx[4] = {0, -1, 0, 1}, dy[4] = {1, 0, -1, 0};
  for (int i = 0; i < 4; i++) {
    if (dfs(board, x + dx[i], y + dy[i], word, pos + 1)) {
      return true;
    }
  }
  board[x][y] = tmp;
  return false;
}

bool exist(std::vector<std::vector<char>> &board, std::string &wold) {
  /*
  int n = board.size(), m = n ? board.at(0).size() : 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (board[i][j] == wold[0]) {
        return dfs(board, i, j, wold, 0);
      }
    }
  }
  return false;
  */

  int n = board.size(), m = n ? board[0].size() : 0;
  std::vector<std::vector<bool>> visited(n, std::vector<bool>(m, false));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {

      if (board[i][j] == wold[0]) {
        std::stack<std::pair<int, int>> s;
        s.push({i, j});
        visited[i][j] = true;
        int p = 1;
        while (!s.empty()) {
          auto pos = s.top();
          s.pop();
          int dx[4] = {0, -1, 0, 1}, dy[4] = {1, 0, -1, 0};
          for (int k = 0; k < 4; k++) {
            int x = pos.first + dx[k], y = pos.second + dy[k];
            if (x < 0 || x >= n || y < 0 || y >= m || visited[x][y] ||
                board[x][y] != wold[p])
              continue;
            if (p == wold.size() - 1)
              return true;
            p++;
            s.push({x, y});
            visited[x][y] = true;
          }
        }
      }
    }
  }
  return false;
}

int main() {
  std::vector<std::vector<char>> board = {
      {'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
  std::string wold = "ABCCED";
  std::cout << "exist: " << exist(board, wold) << std::endl;
  return 0;
}
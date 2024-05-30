#include <iostream>
#include <vector>

/*
给定一个无重复的数字数组[1,2,3]，给出所有的全排列方式
*/

void backtracking(std::vector<int> &nums, std::vector<int> &path,
                  std::vector<bool> &visited,
                  std::vector<std::vector<int>> &ans) {
  if (path.size() == nums.size()) {
    ans.push_back(path);
    return;
  }
  for (int i = 0; i < nums.size(); i++) {
    if (visited[i])
      continue;
    visited[i] = true;
    path.push_back(nums[i]);
    backtracking(nums, path, visited, ans);
    visited[i] = false;
    path.pop_back();
  }
}

std::vector<std::vector<int>> permute(std::vector<int> &nums) {
  std::vector<std::vector<int>> ans;
  std::vector<int> path;
  std::vector<bool> visited(nums.size(), false);
  backtracking(nums, path, visited, ans);
  return ans;
}

int main() {
  std::vector<int> nums = {1, 2, 3};
  auto ans = permute(nums);
  for (int i = 0; i < ans.size(); i++) {
    std::cout << "[";
    for (int j = 0; j < ans[i].size(); j++) {
      std::cout << ans[i][j] << ",";
    }
    std::cout << "]" << std::endl;
  }
  return 0;
}
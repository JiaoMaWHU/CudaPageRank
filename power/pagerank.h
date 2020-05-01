#pragma once
#include <vector>
#include <string>
#include <unordered_map>

int read_file(std::unordered_map<int, std::vector<int> > &m, std::string filename); // read a n*n matrix, return n

int run_pagerank(std::string filename, float epsilon); 
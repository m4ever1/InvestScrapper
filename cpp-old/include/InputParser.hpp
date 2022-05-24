#ifndef __INPUTPARSER_H__
#define __INPUTPARSER_H__

#include <iostream>
#include <vector>
#include <fstream>
#include "Item.hpp"

const static std::string DELIMITER = ":";

class InputParser
{
private:
    std::string myFileName;
    std::vector<Item> myItemVector;
public:
    InputParser(std::string fileName);
    bool buildTransactionsVector(std::vector<std::vector<Item>>& vectorOut);
    ~InputParser();
};


#endif // __INPUTPARSER_H__
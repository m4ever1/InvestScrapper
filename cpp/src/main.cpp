
#include <iostream>
#include "InputParser.hpp"
  
void printTransactions(const std::vector<std::vector<Item>>& transactionsVector)
{
    for(const auto& trans : transactionsVector)
    {
        for(const auto& item : trans)
        {
            std::cout << item.myName << " ";
        }
        std::cout << ":" << std::endl;
        for(const auto& item : trans)
        {
            std::cout << item.myExternalUtil << " ";
        }
        std::cout << ":" << std::endl;
        for(const auto& item : trans)
        {
            std::cout << item.mySector << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    InputParser myInputParser("input.txt");
    
    std::vector<std::vector<Item>> transactionsVector;

    myInputParser.buildTransactionsVector(transactionsVector);


    return 0;
}
#include"InputParser.hpp"
#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>

InputParser::InputParser(std::string fileName) : myFileName(fileName)
{
}

InputParser::~InputParser()
{
}

bool InputParser::buildTransactionsVector(std::vector<std::vector<Item>>& vectorOut)
{
    std::ifstream myfile (myFileName);
    if (myfile.is_open())
    {
        std::string line;
        while ( std::getline (myfile,line) )
        {
            std::vector<Item> itemVectorEntry;
            std::vector<std::string> result;
            //NAMES:PROFITS:SECTORS\n
            boost::split(result, line, boost::is_any_of(DELIMITER));
            if(result.size() > 3)
            {
                std::cout << "Error parsing, can't have more than 2 delimiters per line (:)";
                return false;
            }
            std::vector<std::string> names;
            std::vector<std::string> utils;
            std::vector<std::string> sectors;
            boost::split(names, result[0], boost::is_any_of(" "));
            boost::split(utils, result[1], boost::is_any_of(" "));
            boost::split(sectors, result[2], boost::is_any_of(" "));
            for(int i = 0; i < names.size(); i++)
            {
                Item itemToInsert(names[i], std::stof(utils[i]), sectors[i]);
                itemVectorEntry.push_back(itemToInsert);
            }
            vectorOut.push_back(itemVectorEntry);
        }
        myfile.close();
    }
    else 
    {
        std::cout << "Unable to open file"; 
        return false;
    }
    return true;
}

#include <iostream>
#include "InputParser.hpp"
#include <set>
#include "fptree.hpp"
#include "Utils.hpp"

void test_1()
{
    const Item a1{"A" , 0, "A"};
    const Item b1{"B" , 100, "A"};
    const Item c1{"C" , 57, "A"};
    const Item d1{"D" , 71, "A"};

    const Item a2{"A" , 0, "A"};
    const Item b2{"B" , 43, "A"};
    const Item c2{"C" , 29, "A"};
    const Item d2{"D" , 71, "A"};

    const Item a3{"A" , 43, "A"};
    const Item b3{"B" , 0, "A"};
    const Item c3{"C" , 43, "A"};
    const Item d3{"D" , 43, "A"};

    const Item a4{"A" , 100, "A"};
    const Item b4{"B" , 0, "A"};
    const Item c4{"C" , 43, "A"};
    const Item d4{"D" , 100, "A"};

    const Item a5{"A" , 86, "A"};
    const Item b5{"B" , 71, "A"};
    const Item c5{"C" , 0, "A"};
    const Item d5{"D" , 71, "A"};

    const Item a6{"A" , 57, "A"};
    const Item b6{"B" , 71, "A"};
    const Item c6{"C" , 0, "A"};
    const Item d6{"D" , 71, "A"};

    const std::vector<Transaction> transactions{
        { a1, b1, c1, d1 },
        { a2, b2, c2, d2 },
        { a3, b3, c3, d3 },
        { a4, b4, c4, d4 },
        { a5, b5, c5, d5 },
        { a6, b6, c6, d6 }
    };

    const float minimum_support_threshold = 300;

    Utils utils(transactions, minimum_support_threshold);
    const std::map<Item, float> iwiSupportByItem = utils.getIwiSupportByItem();

    const FPTree fptree(transactions, minimum_support_threshold, iwiSupportByItem);

    const std::set<Pattern> prefix;

    const std::set<Pattern> patterns = utils.IWIMining( fptree, minimum_support_threshold, prefix);

    for(const auto& setfloat : patterns)
    {
        for(const auto& item : setfloat.first)
        {
            std::cout << item.myName << " ";
        }
        std::cout << ": " << setfloat.second << std::endl;
    }

    // assert( patterns.size() == 19 );
    // assert( patterns.count( { { a }, 8 } ) );
    // assert( patterns.count( { { b, a }, 5 } ) );
    // assert( patterns.count( { { b }, 7 } ) );
    // assert( patterns.count( { { c, b }, 5 } ) );
    // assert( patterns.count( { { c, a, b }, 3 } ) );
    // assert( patterns.count( { { c, a }, 4 } ) );
    // assert( patterns.count( { { c }, 6 } ) );
    // assert( patterns.count( { { d, a }, 4 } ) );
    // assert( patterns.count( { { d, c, a }, 2 } ) );
    // assert( patterns.count( { { d, c }, 3 } ) );
    // assert( patterns.count( { { d, b, a }, 2 } ) );
    // assert( patterns.count( { { d, b, c }, 2 } ) );
    // assert( patterns.count( { { d, b }, 3 } ) );
    // assert( patterns.count( { { d }, 5 } ) );
    // assert( patterns.count( { { e, d }, 2 } ) );
    // assert( patterns.count( { { e, c }, 2 } ) );
    // assert( patterns.count( { { e, a, d }, 2 } ) );
    // assert( patterns.count( { { e, a }, 2 } ) );
    // assert( patterns.count( { { e }, 3 } ) );
} 

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
    // std::cout << "starting test" << std::endl;
    
    // test_1();
    InputParser myInputParser("/home/miguel/InvestScrapper/input.txt");
    
    std::vector<Transaction> transactionsVector;

    myInputParser.buildTransactionsVector(transactionsVector);

    const float min_average = 15;

    const float minimum_support_threshold = min_average*transactionsVector.size();

    Utils utils(transactionsVector, minimum_support_threshold);
    const std::map<Item, float> iwiSupportByItem = utils.getIwiSupportByItem();

    const FPTree fptree(transactionsVector, minimum_support_threshold, iwiSupportByItem);

    const std::set<Pattern> prefix;

    const std::set<Pattern> patterns = utils.IWIMining( fptree, minimum_support_threshold, prefix);

    for(const auto& setfloat : patterns)
    {
        for(const auto& item : setfloat.first)
        {
            std::cout << item.myName << " ";
        }
        std::cout << ": " << setfloat.second/transactionsVector.size() << std::endl;
    }

    return 0;
}
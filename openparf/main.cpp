/**
 * @file   main.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/database.h"

int main(int argc, char **argv) {

  openparf::database::Database db(0);

  if (argc < 2) {
    std::cout << "usage: program aux_file\n";
    return 1;
  }

  db.readBookshelf(argv[1]);

  std::cout << db.design() << std::endl;

  return 0;
}

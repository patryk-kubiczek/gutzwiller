cmake_minimum_required(VERSION 3.2)
project(gutzwiller)

set(SOURCE_FILES main.cpp Model.cpp Model.h MultirootSolver.h MultirootSolver.cpp lambda_functions.cpp rho_functions.cpp rho_lambda_eta_functions.cpp model_functions.cpp const_functions.cpp LambdaSolver.cpp RhoSolver.cpp RhoSolver.h GutzwillerSolver.cpp GutzwillerSolver.h CombinedSolver.cpp CombinedSolver.h)
add_executable(gutzwiller ${SOURCE_FILES})

set(LIBS -lm -lgsl -larmadillo)
target_link_libraries(${PROJECT_NAME} ${LIBS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")



/* -----------------------------------------------------------------------------
 *
 *  SPDX-License-Identifier: LGPL-2.1-or-later
 *  Copyright:
 *  2024-2025 by the deal.II authors
 *
 *  This file is part of the blood-flow example built on the deal.II library.
 *  It provides the top-level executable that steers the templated
 *  MetricFlowSystem<1,3> class.  The structure mirrors main_embedded.cc so that
 *  build rules, CMake targets, and user habits remain consistent across
 *  multiple applications in the same repository.
 *
 *  --------------------------------------------------------------------------
 */

#include <deal.II/base/logstream.h> // deallog control
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <stdexcept> // for std::invalid_argument
#include <string>

#include "metric_flow_system.h" // header exposing MetricFlowSystem

using namespace dealii;

namespace
{
  class CommandLineError : public std::runtime_error
  {
  public:
    explicit CommandLineError(const std::string &message)
      : std::runtime_error(message)
    {}
  };

  enum class CommandLineMode
  {
    run,
    print_parameters,
    validate_parameters
  };

  struct CommandLine
  {
    CommandLineMode mode = CommandLineMode::run;
    std::string     parameter_file;
  };

  void
  print_usage(std::ostream &out, const char *program)
  {
    out
      << "Usage:\n"
      << "  " << program << " PARAMETER_FILE\n"
      << "  " << program << " --help\n"
      << "  " << program << " --print-parameters\n"
      << "  " << program << " --validate-parameters PARAMETER_FILE\n\n"
      << "Options:\n"
      << "  -h, --help                         Show this help text.\n"
      << "  --print-parameters                 Print the registered parameter schema\n"
      << "                                      and its existing defaults.\n"
      << "  --validate-parameters FILE         Parse FILE against the registered\n"
      << "                                      ParameterHandler schema only.\n\n"
      << "A normal run requires an existing parameter file; no default file is\n"
      << "created when the argument is omitted or does not exist.\n";
  }

  bool
  is_option(const std::string &argument)
  {
    return argument == "-h" || argument == "--help" ||
           argument == "--print-parameters" ||
           argument == "--validate-parameters";
  }

  CommandLine
  parse_command_line(const int argc, char **argv)
  {
    if (argc == 1)
      throw CommandLineError("a parameter file is required for a normal run");

    const std::string first_argument(argv[1]);
    if (first_argument == "-h" || first_argument == "--help")
      {
        if (argc != 2)
          throw CommandLineError("--help does not accept additional arguments");
        return {CommandLineMode::run, ""};
      }

    if (first_argument == "--print-parameters")
      {
        if (argc != 2)
          throw CommandLineError(
            "--print-parameters does not accept additional arguments");
        return {CommandLineMode::print_parameters, ""};
      }

    if (first_argument == "--validate-parameters")
      {
        if (argc != 3 || is_option(argv[2]))
          throw CommandLineError(
            "--validate-parameters requires exactly one parameter file");
        return {CommandLineMode::validate_parameters, argv[2]};
      }

    if (!first_argument.empty() && first_argument.front() == '-')
      throw CommandLineError("unknown option: " + first_argument);

    if (argc != 2)
      throw CommandLineError("a normal run accepts exactly one parameter file");

    return {CommandLineMode::run, first_argument};
  }

  void
  require_readable_parameter_file(const std::string &filename)
  {
    std::ifstream parameter_file(filename);
    if (!parameter_file)
      throw CommandLineError("cannot open parameter file: " + filename);
  }
} // namespace

int
main(int argc, char **argv)
{
  try
    {
      const CommandLine command_line = parse_command_line(argc, argv);

      if (command_line.mode == CommandLineMode::run &&
          command_line.parameter_file.empty())
        {
          print_usage(std::cout, argv[0]);
          return 0;
        }

      // The executable owns its command-line grammar.  Do not pass application
      // options or parameter paths to deal.II/PETSc, which would otherwise
      // report them as unused database options before this program handles
      // them.
      int                              mpi_argc           = 1;
      char                            *mpi_argv_storage[] = {argv[0], nullptr};
      char                           **mpi_argv           = mpi_argv_storage;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(mpi_argc,
                                                          mpi_argv,
                                                          1);

      if (command_line.mode == CommandLineMode::run ||
          command_line.mode == CommandLineMode::validate_parameters)
        require_readable_parameter_file(command_line.parameter_file);

      MetricFlowSystem<1, 3> problem; // 1-dim geometry embedded in \mathbb{R}^3

      if (command_line.mode == CommandLineMode::print_parameters)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            ParameterAcceptor::prm.print_parameters(std::cout,
                                                    ParameterHandler::PRM);
          return 0;
        }

      if (command_line.mode == CommandLineMode::validate_parameters)
        {
          // An empty output filename makes validation read-only.  The explicit
          // existence check above prevents ParameterAcceptor from creating a
          // missing input file as part of its normal error handling.
          ParameterAcceptor::initialize(command_line.parameter_file,
                                        "",
                                        ParameterHandler::Short,
                                        ParameterAcceptor::prm,
                                        ParameterHandler::Short);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Parameter file is valid: "
                      << command_line.parameter_file << '\n';
          return 0;
        }

      /* ---------------------- 1. Initialise deal.II logging -----------------
       */
      dealii::deallog.depth_console(1);

      /* ------------------------- 2. Set up the problem ----------------------
       */
      problem.initialize_params(command_line.parameter_file);
      problem.run();

      return 0;
    }
  catch (const CommandLineError &error)
    {
      std::cerr << "Command-line error: " << error.what() << "\n\n";
      print_usage(std::cerr, argv[0]);
      return 2;
    }
  catch (const std::invalid_argument &theta_range) // parameter-specific errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Invalid parameter: \n"
                << theta_range.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (const std::exception &exc) // general deal.II/runtime errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Exception on processing: \n"
                << exc.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...) // fallback
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Unknown exception!\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
}

/* $Id: scanner.ll 44 2008-10-23 09:03:19Z tb $ -*- mode: c++ -*- */
/** \file scanner.ll Define the example Flex lexical scanner */

%{ /*** C/C++ Declarations ***/

#include <string>

#include "bookshelf_scanner.h"

/* import the parser's token type into a local typedef */
typedef bookshelfparser::Parser::token token;
typedef bookshelfparser::Parser::token_type token_type;

/* By default yylex returns int, we use token_type. Unfortunately yyterminate
 * by default returns 0, which is not of token_type. */
#define yyterminate() return token::ENDF

/* This disables inclusion of unistd.h, which is not available under Visual C++
 * on Win32. The C++ scanner uses STL streams instead. */
#define YY_NO_UNISTD_H

%}

/*** Flex Declarations and Options ***/

/* enable c++ scanner class generation */
%option c++

/* change the name of the scanner class. results in "ExampleFlexLexer" */
%option prefix="BookshelfParser"

/* the manual says "somewhat more optimized" */
%option batch

/* enable scanner to generate debug output. disable this for release
 * versions. */
%option debug

/* no support for include files is planned */
%option yywrap nounput 

/* enables the use of start condition stacks */
%option stack

/* The following paragraph suffices to track locations accurately. Each time
 * yylex is invoked, the begin position is moved onto the end position. */
%{
#define YY_USER_ACTION  yylloc->columns(yyleng);
%}

%% /*** Regular Expressions Part ***/

 /* code to place at the beginning of yylex() */
%{
    // reset location
    yylloc->step();
%}

 /*** BEGIN EXAMPLE - Change the example lexer rules below ***/

(?i:SITE)                    { return token::KWD_SITE; }
(?i:END)                     { return token::KWD_END; }
(?i:RESOURCES)               { return token::KWD_RESOURCES; }
(?i:SITEMAP)                 { return token::KWD_SITEMAP; }
(?i:SLICEL)                  { return token::KWD_SLICEL; }
(?i:SLICEM)                  { return token::KWD_SLICEM; }
(?i:SLICE)                   { return token::KWD_SLICE; }
(?i:DSP)                     { return token::KWD_DSP; }
(?i:BRAM)                    { return token::KWD_BRAM; }
(?i:IO)                      { return token::KWD_IO; }
(?i:INPUT)                   { return token::KWD_INPUT; }
(?i:OUTPUT)                  { return token::KWD_OUTPUT; }
(?i:CLOCK)                   { return token::KWD_CLOCK; }
(?i:CTRL_SR)                 { return token::KWD_CTRL_SR; }
(?i:CTRL_CE)                 { return token::KWD_CTRL_CE; }
(?i:CTRL)                    { return token::KWD_CTRL; }
(?i:CAS)                     { return token::KWD_CAS; }
(?i:FIXED)                   { return token::KWD_FIXED; }
(?i:CELL)                    { return token::KWD_CELL; }
(?i:PAR)                     { return token::KWD_PAR; }
(?i:PIN)                     { return token::KWD_PIN; }
(?i:net)                     { return token::KWD_NET; }
(?i:endnet)                  { return token::KWD_ENDNET; }
(?i:CLOCKREGION)             { return token::KWD_CLOCKREGION; }
(?i:CLOCKREGIONS)            { return token::KWD_CLOCKREGIONS; }
(?i:Shape)                   { return token::KWD_SHAPE; }
(?i:Type)                    { return token::KWD_TYPE; }

[A-Za-z0-9_]+\.lib           { yylval->strVal = new std::string(yytext, yyleng); return token::LIB_FILE; }
[A-Za-z0-9_]+\.scl           { yylval->strVal = new std::string(yytext, yyleng); return token::SCL_FILE; }
[A-Za-z0-9_]+\.nodes         { yylval->strVal = new std::string(yytext, yyleng); return token::NODE_FILE; }
[A-Za-z0-9_]+\.nets          { yylval->strVal = new std::string(yytext, yyleng); return token::NET_FILE; }
[A-Za-z0-9_]+\.pl            { yylval->strVal = new std::string(yytext, yyleng); return token::PL_FILE; }
[A-Za-z0-9_]+\.wts           { yylval->strVal = new std::string(yytext, yyleng); return token::WT_FILE; }
[A-Za-z0-9_]+\.shapes        { yylval->strVal = new std::string(yytext, yyleng); return token::SHAPE_FILE; }

[0-9]+[']b[0-1]+    {
                    char* end;
                    yylval->mask.bits  = strtol(yytext,&end,10);
                    yylval->mask.value = strtol(end+2*sizeof(char),NULL,2); 
                    return token::BIT_MASK;
}
[0-9]+[']o[0-7]+    {
                    char* end;
                    yylval->mask.bits  = strtol(yytext,&end,10);
                    yylval->mask.value = strtol(end+2*sizeof(char),NULL,8); 
                    return token::OCT_MASK;
}
[0-9]+[']d[0-9]+    {
                    char* end;
                    yylval->mask.bits  = strtol(yytext,&end,10);
                    yylval->mask.value = strtol(end+2*sizeof(char),NULL,10); 
                    return token::DEC_MASK;
}
[0-9]+[']h[0-9a-fA-F]+  {
                    char* end;
                    yylval->mask.bits  = strtol(yytext,&end,10);
                    yylval->mask.value = strtol(end+2*sizeof(char),NULL,16); 
                    return token::HEX_MASK;
}

[\+\-]?[0-9]+                { yylval->intVal = atol(yytext); return token::INT; }
[\+\-]?[0-9]+\.[0-9]+        { yylval->dblVal = atof(yytext); return token::DOUBLE; }
[A-Za-z_\\][A-Za-z0-9_/\[\]\-\\<>]* { yylval->strVal = new std::string(yytext, yyleng); return token::STRING; }

 /* gobble up comments */
"#"[^\n]*                    { yylloc->step(); }

 /* gobble up white-spaces */
[ \t\r]+                     { yylloc->step(); }

 /* gobble up end-of-lines */
\n                           { yylloc->lines(yyleng); yylloc->step(); return token::ENDL; }

 /* pass all other characters up to bison */
.                            { return static_cast<token_type>(*yytext); }

 /*** END EXAMPLE - Change the example lexer rules above ***/

%% /*** Additional Code ***/

namespace bookshelfparser {

Scanner::Scanner(std::istream* in, std::ostream* out)
    : BookshelfParserFlexLexer(in, out)
{}

Scanner::~Scanner() {}

void Scanner::set_debug(bool b) { yy_flex_debug = b; }

}

/* This implementation of ExampleFlexLexer::yylex() is required to fill the
 * vtable of the class ExampleFlexLexer. We define the scanner's main yylex
 * function via YY_DECL to reside in the Scanner class instead. */

#ifdef yylex
#undef yylex
#endif

int BookshelfParserFlexLexer::yylex()
{
    std::cerr << "in BookshelfParserFlexLexer::yylex() !" << std::endl;
    return 0;
}

/* When the scanner receives an end-of-file indication from YY_INPUT, it then
 * checks the yywrap() function. If yywrap() returns false (zero), then it is
 * assumed that the function has gone ahead and set up `yyin' to point to
 * another input file, and scanning continues. If it returns true (non-zero),
 * then the scanner terminates, returning 0 to its caller. */

int BookshelfParserFlexLexer::yywrap()
{
    return 1;
}

/* $Id: parser.yy 48 2009-09-05 08:07:10Z tb $ -*- mode: c++ -*- */
/** \file parser.yy Contains the example Bison parser source */

%{ /*** C/C++ Declarations ***/

#include <stdio.h>
#include <string>
#include <vector>
#include <array>

%}

/*** yacc/bison Declarations ***/

/* Require bison 2.3 or later */
%require "2.3"

/* add debug output code to generated parser. disable this for release
 * versions. */
%debug

/* start symbol is named "start" */
/*%start start*/

/* write out a header file containing the token defines */
%defines

/* use newer C++ skeleton file */
%skeleton "lalr1.cc"

/* namespace to enclose parser in */
%define api.namespace {bookshelfparser}

/* set the parser's class identifier */
%define api.parser.class {Parser}

/* keep track of the current position within the input */
%locations
%initial-action
{
    // initialize the initial location object
    @$.begin.filename = @$.end.filename = &driver.streamname;
};

/* The driver is passed by reference to the parser and to the scanner. This
 * provides a simple but effective pure interface, not relying on global
 * variables. */
%parse-param { class Driver& driver }

/* verbose error messages */
%define parse.error verbose

 /*** BEGIN EXAMPLE - Change the example grammar's tokens below ***/

%union {
    unsigned      intVal;
    double        dblVal;
    struct {
        long value;
        long bits;
    } mask;
    std::string*  strVal;
}

%token                ENDF       0  "end of file"
%token                ENDL
%token                BIT_MASK;
%token                OCT_MASK;
%token                DEC_MASK;
%token                HEX_MASK;
%token    <intVal>    INT
%token    <dblVal>    DOUBLE
%token    <strVal>    STRING
%token                KWD_END
%token                KWD_SITEMAP
%token                KWD_SLICEL
%token                KWD_SLICEM
%token                KWD_SLICE
%token                KWD_DSP
%token                KWD_BRAM
%token                KWD_IO
%token                KWD_SITE
%token                KWD_RESOURCES
%token                KWD_FIXED
%token                KWD_CELL
%token                KWD_PAR
%token                KWD_PIN
%token                KWD_INPUT
%token                KWD_OUTPUT
%token                KWD_CLOCK
%token                KWD_CTRL
%token                KWD_CTRL_SR
%token                KWD_CTRL_CE
%token                KWD_CAS
%token                KWD_NET
%token                KWD_ENDNET
%token                KWD_CLOCKREGION
%token                KWD_CLOCKREGIONS
%token                KWD_SHAPE
%token                KWD_TYPE

%token    <strVal>    LIB_FILE
%token    <strVal>    SCL_FILE
%token    <strVal>    NODE_FILE
%token    <strVal>    NET_FILE
%token    <strVal>    PL_FILE
%token    <strVal>    WT_FILE
%token    <strVal>    SHAPE_FILE

%type<mask> BIT_MASK OCT_MASK DEC_MASK HEX_MASK

%destructor { delete $$; } STRING

 /*** END EXAMPLE - Change the example grammar's tokens above ***/

%{

#include "bookshelf_driver.h"
#include "bookshelf_scanner.h"

/* this "connects" the bison parser in the driver to the flex scanner class
 * object. it defines the yylex() function call to pull the next token from the
 * current lexer object of the driver context. */
#undef yylex
#define yylex driver.scanner->lex

%}

%% /*** Grammar Rules ***/

/***** top patterns *****/
top : ENDL_STAR sub_top
    ;

sub_top : aux_top
        | lib_top
        | scl_top
        | node_top
        | pl_top
        | net_top
        | wt_top
        | shape_top
        ;

/***** aux file *****/
aux_top : aux_line
        ;

aux_line : STRING ':' aux_files ENDLS { delete $1; }
         ;

aux_files : aux_files aux_file
          | aux_file
          ;

aux_file : LIB_FILE   { driver.setLibFileCbk(*$1); delete $1; }
         | SCL_FILE   { driver.setSclFileCbk(*$1); delete $1; }
         | NODE_FILE  { driver.setNodeFileCbk(*$1); delete $1; }
         | NET_FILE   { driver.setNetFileCbk(*$1); delete $1; }
         | PL_FILE    { driver.setPlFileCbk(*$1); delete $1; }
         | WT_FILE    { driver.setWtFileCbk(*$1); delete $1; }
         | SHAPE_FILE { driver.setShapeFileCbk(*$1); delete $1; }
         ;


/***** lib file *****/
lib_top : cell_blocks
        ;

cell_blocks : cell_blocks cell_block
            | cell_block
            ;

cell_block : cell_block_header
             cell_block_lines
             cell_block_footer
           ;

cell_block_header : KWD_CELL STRING ENDL { driver.addCellCbk(*$2); delete $2; }
                  ;

cell_block_footer : KWD_END KWD_CELL ENDL_STAR
                  ;

cell_block_lines  : cell_block_lines cell_block_line
                  | cell_block_line
                  ;

cell_block_line : KWD_PIN STRING KWD_INPUT ENDL             { driver.addCellInputPinCbk(*$2);    delete $2; }
                | KWD_PIN STRING KWD_OUTPUT ENDL            { driver.addCellOutputPinCbk(*$2);   delete $2; }
                | KWD_PIN STRING KWD_INPUT KWD_CLOCK ENDL   { driver.addCellClockPinCbk(*$2);    delete $2; }
                | KWD_PIN STRING KWD_INPUT KWD_CTRL ENDL    { driver.addCellCtrlPinCbk(*$2);     delete $2; }
                | KWD_PIN STRING KWD_INPUT KWD_CTRL_SR ENDL { driver.addCellCtrlSRPinCbk(*$2);   delete $2; }
                | KWD_PIN STRING KWD_INPUT KWD_CTRL_CE ENDL { driver.addCellCtrlCEPinCbk(*$2);   delete $2; }
                | KWD_PIN STRING KWD_INPUT KWD_CAS ENDL     { driver.addCellInputCasPinCbk(*$2); delete $2; }
                | KWD_PIN STRING KWD_OUTPUT KWD_CAS ENDL    { driver.addCellInputCasPinCbk(*$2); delete $2; }
                | KWD_PAR STRING ENDL                       { driver.addCellParameterCbk(*$2);   delete $2; }
                ;


/*! scl file
 *  Clock region block is optional, since ISPD16 benchmark doesn't have clock regions
 */
scl_top : site_blocks rsrc_block sitemap_block
        | site_blocks rsrc_block sitemap_block clock_region_block
        ;

/* site blocks */
site_blocks : site_blocks site_block
            | site_block
            ;

site_block : site_block_header
             site_block_lines
             site_block_footer
             | site_block_header 
             site_block_footer
           ;

site_block_header : KWD_SITE site_type_name ENDL
                  ;

site_type_name : KWD_SLICEL { driver.setSiteTypeCbk("SLICEL"); }
               | KWD_SLICEM { driver.setSiteTypeCbk("SLICEM"); }
               | KWD_SLICE { driver.setSiteTypeCbk("SLICE"); } 
               | KWD_DSP { driver.setSiteTypeCbk("DSP"); }
               | KWD_BRAM { driver.setSiteTypeCbk("BRAM"); }
               | KWD_IO { driver.setSiteTypeCbk("IO"); }
               | STRING { driver.setSiteTypeCbk(*$1); delete $1; }
               ;

site_block_footer : KWD_END KWD_SITE ENDL_STAR { driver.endSiteBlockCbk(); }
                  ;

site_block_lines : site_block_lines site_block_line
                 | site_block_line
                 ;

site_block_line : rsrc_type_name INT ENDL { driver.setSiteNumResourcesCbk($2); }
                ;

rsrc_type_name : STRING { driver.setResourceTypeCbk(*$1); delete $1; }
               | KWD_IO { driver.setResourceTypeCbk("IO"); }
               | KWD_BRAM { driver.setResourceTypeCbk("BRAM"); }
               ;

/* resources block */
rsrc_block : rsrc_block_header
             rsrc_block_lines
             rsrc_block_footer
           ;

rsrc_block_header : KWD_RESOURCES ENDL
                  ;

rsrc_block_footer : KWD_END KWD_RESOURCES ENDL_STAR { driver.endResourceTypeBlockCbk(); }
                  ;

rsrc_block_lines : rsrc_block_lines rsrc_block_line
                 | rsrc_block_line
                 ;

rsrc_block_line : rsrc_type_name cell_name_list ENDL {
                driver.addResourceTypeCbk(); 
                }
                ;

cell_name_list : cell_name_list STRING { driver.addToCellNameListCbk(*$2); delete $2; }
               | STRING { driver.addToCellNameListCbk(*$1); delete $1; }
               ;

/* sitemap block */
sitemap_block : sitemap_block_header
                sitemap_block_lines
                sitemap_block_footer
              ;

sitemap_block_header : KWD_SITEMAP INT INT ENDL { driver.initSiteMapCbk($2, $3); };

sitemap_block_footer : KWD_END KWD_SITEMAP ENDL_STAR { driver.endSiteMapCbk(); }
                     ;

sitemap_block_lines : sitemap_block_lines sitemap_block_line
                    | sitemap_block_line
                    ;

sitemap_block_line : INT INT KWD_SLICEL ENDL { driver.setSiteMapEntryCbk($1, $2, "SLICEL"); }
                   | INT INT KWD_SLICEM ENDL { driver.setSiteMapEntryCbk($1, $2, "SLICEM"); }
                   | INT INT KWD_SLICE ENDL { driver.setSiteMapEntryCbk($1, $2, "SLICE"); }
                   | INT INT KWD_DSP ENDL    { driver.setSiteMapEntryCbk($1, $2, "DSP"); }
                   | INT INT KWD_BRAM ENDL   { driver.setSiteMapEntryCbk($1, $2, "BRAM"); }
                   | INT INT KWD_IO ENDL     { driver.setSiteMapEntryCbk($1, $2, "IO"); }
                   | INT INT STRING ENDL     { driver.setSiteMapEntryCbk($1, $2, *$3); delete $3; }
                   ;

/* clock region block */
clock_region_block : clock_region_block_header
                     clock_region_block_lines
                     clock_region_block_footer
                   ;

clock_region_block_header : KWD_CLOCKREGIONS INT INT ENDL { driver.initClockRegionsCbk($2, $3); }
                          ;

clock_region_block_footer : KWD_END KWD_CLOCKREGIONS ENDL_STAR
                          ;

clock_region_block_lines : clock_region_block_lines clock_region_block_line
                         | clock_region_block_line
                         ;

clock_region_block_line : KWD_CLOCKREGION STRING ':' INT INT INT INT INT INT ENDL { driver.addClockRegionCbk(*$2, $4, $5, $6, $7, $8, $9); delete $2; }
                        ;


/***** node file *****/
node_top : node_lines
         ;

node_lines : node_lines node_line
           | node_line
           ;

node_entry : STRING STRING { driver.addNodeCbk(*$1, *$2); delete $1; delete $2; }
           | node_entry BIT_MASK
           | node_entry OCT_MASK 
           | node_entry DEC_MASK 
           | node_entry HEX_MASK 
          ; 

node_line : node_entry ENDL_STAR 
          ;

/***** pl file *****/
pl_top : pl_lines
          ;

pl_lines : pl_lines pl_line
         | pl_line
         ;

pl_line : STRING INT INT INT KWD_FIXED ENDL_STAR { driver.setFixedNodeCbk(*$1, $2, $3, $4); delete $1; }
        ;

/***** net file *****/
net_top : net_blocks
        ;

net_blocks : net_blocks net_block
           | net_block
           ;

net_block : net_block_header
            net_block_lines
            net_block_footer
          ;

net_block_header : KWD_NET STRING INT ENDL { driver.addNetCbk(*$2, $3); delete $2; }
                 ;

net_block_footer : KWD_ENDNET ENDL_STAR
                 ;

net_block_lines : net_block_lines net_block_line
                | net_block_line
                ;

net_block_line : STRING STRING ENDL { driver.addPinCbk(*$1, *$2); delete $1; delete $2; }
               ;

/***** wt file *****/
wt_top : ENDL_STAR
       ;


/* swallow ENDL by recursion */
ENDLS : ENDLS ENDL
      | ENDL
      ;

ENDL_STAR : ENDLS
          | /* empty */
          ;

/***** shape file *****/
shape_top : shape_blocks
          ;

shape_blocks : shape_blocks shape_block
             | shape_block
             ;

shape_block : shape_block_header
              shape_block_type_line
              shape_block_node_lines
              shape_block_footer
            ;

shape_block_header : KWD_SHAPE INT '{' ENDL
                   ;

shape_block_footer : '}' ENDL_STAR
                   ;

shape_block_type_line : KWD_TYPE STRING ENDL        { driver.addShapeCbk(*$2); delete $2; }
                      ;

shape_block_node_lines : shape_block_node_lines shape_block_node_line
                       | shape_block_node_line
                       ;

shape_block_node_line : INT INT STRING STRING ENDL { driver.addShapeNodeCbk($1, $2, *$4); delete $3; delete $4; }
                      ;


/*** END EXAMPLE - Change the example grammar rules above ***/

%% /*** Additional Code ***/

void bookshelfparser::Parser::error(const Parser::location_type& l, const std::string& m)
{
    driver.error(l, m);
    exit(1);
}

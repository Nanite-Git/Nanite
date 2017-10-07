//+------------------------------------------------------------------+
//|                                                      Test001.mq5 |
//|                                                          HolgDer |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "HolgDer"
#property link      ""
#property version   "1.00"
//--- input parameters
#include <MQLMySQL.mqh>
#include <SQL_CONF.mqh>
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
const string name = "nanite-holg";
string tab_name = "HistoryData";
string Errorcode = "";

int OnInit()
  {
   SQL_CONF sqlc(name);                   // creating new object to connect to sql database 
   sqlc.initialize_db();                  // initialize recommended values
   sqlc.connect_db();                     // connecting to database
   //------------------<CREATE A TABLE>-----------------------------//
   sqlc.create_table(tab_name, Errorcode, 1); 
   
   sqlc.disconnect_db();                  // deinitialize the connection
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+

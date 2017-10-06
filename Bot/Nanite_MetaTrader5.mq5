//+------------------------------------------------------------------+
//|                                           Nanite_MetaTrader5.mq5 |
//|                                                          HolgDer |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Nanite"
#property link      ""
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert INCLUDES                                                  |
//+------------------------------------------------------------------+
#include <lib_cisnewbar.mqh>
#include <MQLMySQL.mqh>
#include <SQL_CONF.mqh>
//+------------------------------------------------------------------+
//| Expert GLOBAL DECLARATIONS g_[...] = global                                 |
//+------------------------------------------------------------------+
CisNewBar         g_current_chart;     // Instance of the CisNewBar class
datetime          g_start_time;        // Start time from where to copy the history data
string            g_symbol;            // Symbol on wich this program is used to. For instance: EUR/USD
ENUM_TIMEFRAMES   g_period;            // Period on wich this program is used to. For instance: 10 min
int               g_new_bars;          // Number of new bars
string            g_comment;           // Comment of executing method, associated with current class instance
string            g_Errorcode;         // self descriptive
string            g_tab_name;          // SQL-Table Name
//---
datetime          t_actual;            // current Date and Time
datetime          t_start;             // Start time from where to copy the History data
datetime          t_stop;              // Unitl here the History Data will be copied
//---
bool              newDataFlag;         // Write this Flag to the SQL-Database to indicate that there is new History Data added
string            g_flag_tab_name;     // Name of the SQL-Flag Table (here: flag)
//---Pre-Definition---
void comSQL();
void recNewBar();
string GetPeriodName(ENUM_TIMEFRAMES);
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//---
   printf("Start Init...");
//---   
   g_tab_name = "HistoryData";
   g_flag_tab_name = "new_bar_flag";
   g_Errorcode = "";
//---   
   t_actual = TimeCurrent();
   t_start = D'2017.09.12 13:00:00';
   t_stop = t_actual;
//---Parametrize the function OnTimer() with x Seconds periodically recall
   EventSetTimer(1);
//---
   printf("...Init Finished");
//---
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//---
   EventKillTimer();
}
//+------------------------------------------------------------------+
//| Expert timer function                                            |
//| Pulling new Bar periodically                                     |
//+------------------------------------------------------------------+
void OnTimer()
{
//---
   recNewBar();      //recognize new Bar
}
//+------------------------------------------------------------------+
//| Trade function (not in use)                                      |
//+------------------------------------------------------------------+
void OnTrade()
{
//---
}
//+------------------------------------------------------------------+
//| Communication to SQL-Database                                    |
//| Establish the communication to the SQL Database:                 |
//| see -> SQL_CONF.mqh                                              |
//+------------------------------------------------------------------+
void comSQL()
{
//---new Data are availible -> so set the newData-Flag
   newDataFlag = true;
//---Parametrize the Start and Stop Time to copy One the Bar of the last 60 seconds
//---Here we used 60 seconds to copy a 1Min-Period Bar
   t_start = TimeCurrent()-60;      //Change the 60 Seconds to 600 Seconds if the Period is 10Min
   t_stop = TimeCurrent();
   
   printf("Start: %i",t_start);
   printf("Stop: %i",t_stop);
   
   int r_ch;                              // returnvalue counter number of history-values
   MqlRates hist_data[];                  // stored History data (Cose, High, Low, Open, Volume, ...)
   const string name = "nanite-holg";
   //------------------<CONNECT TO DB and INIT>---------------------//
   SQL_CONF sqlc(name);                   // creating new object to connect to sql database 
   sqlc.initialize_db();                  // initialize recommended values
   sqlc.connect_db();                     // connecting to database
   //------------------<CREATE A TABLE>-----------------------------//
   //sqlc.create_table(tab_name, Errorcode, 1); 
   //----------------------<GET HISTORY DATA>-----------------------//
   r_ch = CopyRates(Symbol(),PERIOD_M1,t_start,t_stop,hist_data);   // get histor values in 'hist_data'   
   printf("Nr of Bars wich will be written to Database: %i", r_ch);
   if(r_ch == -1) {
      Print("Error 'Copy Rates, r_ch = -1'");
   }
   else {
      Print("Successfully copied Hist-Data from choosen Symbol: ", Symbol());
   }
   //--------------------<WRITE HIST DATA TO DB>--------------------//
   sqlc.write_db(hist_data,r_ch,g_tab_name);  
   sqlc.set_newData_flag(g_flag_tab_name, newDataFlag, g_Errorcode);       // Errorcode is currently not evaluated
   //--------------------<DISCONNECT FROM DB>-----------------------//
   sqlc.disconnect_db();                  // deinitialize the connection
}
//+------------------------------------------------------------------+
//| New Bar recognizer                                               |
//+------------------------------------------------------------------+
void recNewBar()
{
//---
   //--- Examine the current_chart instance of class:
   g_symbol = g_current_chart.GetSymbol();       // Get chart symbol, associated with current class instance
   g_period = g_current_chart.GetPeriod();       // Get chart period, associated with current class instance
   
   if(g_current_chart.isNewBar())                // Make request for new bar using the isNewBar() method, associated with current class instance     
   {
      g_comment=g_current_chart.GetComment();    // Get comment of executing method, associated with current class instance
      g_new_bars = g_current_chart.GetNewBars(); // Get number of new bars, associated with current class instance
      Print(g_symbol,GetPeriodName(g_period),g_comment," Number of new bars = ",g_new_bars," Time = ",TimeToString(TimeCurrent(),TIME_SECONDS));
      comSQL();
   }
   else 
   {
      uint error=g_current_chart.GetRetCode();   // Get code of error, associated with current class instance
      if(error!=0)
      {
         g_comment=g_current_chart.GetComment(); // Get comment of executing method, associated with current class instance
         Print(g_symbol,GetPeriodName(g_period),g_comment," Error ",error," Time = ",TimeToString(TimeCurrent(),TIME_SECONDS));
      }
   }  
} 
//+------------------------------------------------------------------+
//| returns string value of the period                               |
//+------------------------------------------------------------------+
string GetPeriodName(ENUM_TIMEFRAMES period)
{
   if(g_period==PERIOD_CURRENT) g_period=Period();
//---
   switch(g_period)
     {
      case PERIOD_M1:  return(" M1 ");
      case PERIOD_M2:  return(" M2 ");
      case PERIOD_M3:  return(" M3 ");
      case PERIOD_M4:  return(" M4 ");
      case PERIOD_M5:  return(" M5 ");
      case PERIOD_M6:  return(" M6 ");
      case PERIOD_M10: return(" M10 ");
      case PERIOD_M12: return(" M12 ");
      case PERIOD_M15: return(" M15 ");
      case PERIOD_M20: return(" M20 ");
      case PERIOD_M30: return(" M30 ");
      case PERIOD_H1:  return(" H1 ");
      case PERIOD_H2:  return(" H2 ");
      case PERIOD_H3:  return(" H3 ");
      case PERIOD_H4:  return(" H4 ");
      case PERIOD_H6:  return(" H6 ");
      case PERIOD_H8:  return(" H8 ");
      case PERIOD_H12: return(" H12 ");
      case PERIOD_D1:  return(" Daily ");
      case PERIOD_W1:  return(" Weekly ");
      case PERIOD_MN1: return(" Monthly ");
     }
//---
   return("unknown period");
} 
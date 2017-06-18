//+------------------------------------------------------------------+
//|                                                     SmartBot.mq5 |
//|                                                           Holger |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Holger"
#property link      ""
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

#include <MQLMySQL.mqh>
#include "SQL_CONF.mqh"

string INI;

void OnStart()
{
   const string name = "nanite-holg";
   SQL_CONF sql_connector(name);
   
   sql_connector.initialize_db();

/*
 string Host, User, Password, Database, Socket, act_symbol="EURUSD"; // database credentials
 int Port,ClientFlag;
 int DB; // database identifier
 
 //-----------< Hist Data declarations >----------------
 int r_cc = 0;
 datetime t_start = D'1990.01.01 00:00:00', t_stop = D'1990.04.01 00:00:00';
 MqlRates hist_data[];
 
 Print (MySqlVersion());

 INI = TerminalInfoString(TERMINAL_PATH)+"\\MQL5\\Scripts\\MyConnection.ini";
 Print(INI);
 
 // reading database credentials from INI file
 Host = ReadIni(INI, "MYSQL", "Host");
 User = ReadIni(INI, "MYSQL", "User");
 Password = ReadIni(INI, "MYSQL", "Password");
 Database = ReadIni(INI, "MYSQL", "Database");
 Port     = (int)StringToInteger(ReadIni(INI, "MYSQL", "Port"));
 Socket   = ReadIni(INI, "MYSQL", "Socket");
 ClientFlag = CLIENT_MULTI_STATEMENTS; //(int)StringToInteger(ReadIni(INI, "MYSQL", "ClientFlag"));  

 Print ("Host: ",Host, ", User: ", User, ", Database: ",Database);
 
 // open database connection
 Print ("Connecting...");
 
 DB = MySqlConnect(Host, User, Password, Database, Port, Socket, ClientFlag);
 
 if (DB == -1) { Print ("Connection failed! Error: "+MySqlErrorDescription); } else { Print ("Connected! DBID#",DB);}
 
 string Query;
 Query = "DROP TABLE IF EXISTS `test_table`";
 
 if(!MySqlExecute(DB, Query)) {
   Print(MySqlErrorDescription,"\n");
 }
 
 
 
 
 //----------------------< GET HISTORY DATA >------------------------
 r_cc = CopyRates(act_symbol,PERIOD_D1,t_start,t_stop,hist_data);
   
   if(r_cc == -1) {
      Print("Error 'Copy Rates'");
   }
 //------------------------------------------------------------------
 
 
 
 Query = "CREATE TABLE `test_table` (id int, code varchar(50), start_date datetime, closevalue double, openvalue double, highvalue double, lowvalue double)";
 if (MySqlExecute(DB, Query)) {
   Print ("Table `test_table` created.");
     
     // Inserting data 1 row
     /*Query = "INSERT INTO `test_table` (id, code, start_date) VALUES ("+(string)AccountInfoInteger(ACCOUNT_LOGIN)+",\'ACCOUNT\',\'"+TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS)+"\')";
     if (MySqlExecute(DB, Query))
        {
         Print ("Succeeded: ", Query);
        }
     else
        {
         Print ("Error: ", MySqlErrorDescription);
         Print ("Query: ", Query);
        }
     */
     // multi-insert
 /*    
 
   for(int count = 0; count < r_cc-1 ; count++) {
      Print(TimeToString(hist_data[count].time)," - ",hist_data[count].close);
      
     Query = "INSERT INTO `test_table` (id, code, start_date, closevalue, openvalue, highvalue, lowvalue) VALUES ("+
                                                                                    (string)count+",\'EURUSD\',\'"+
                                                                                    (string)hist_data[count].time+"\',\'"+
                                                                                    (string)hist_data[count].close+"\',\'"+
                                                                                    (string)hist_data[count].open+"\',\'"+
                                                                                    (string)hist_data[count].high+"\',\'"+
                                                                                    (string)hist_data[count].low+"\');";
     Print(Query);
     
     if (MySqlExecute(DB, Query))
        {
         Print ("Succeeded! 3 rows has been inserted by one query.");
        }
     else
        {
         Print ("Error of multiple statements: ", MySqlErrorDescription);
        }
   }
 }
 else
    {
     Print ("Table `test_table` cannot be created. Error: ", MySqlErrorDescription);
    }
 
 MySqlDisconnect(DB);
 Print ("Disconnected. Script done!");
 */
}
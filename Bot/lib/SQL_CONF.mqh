//+------------------------------------------------------------------+
//|                                                     SQL_CONF.mqh |
//|                                                          HolgDer |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "HolgDer"
#property link      ""
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include <MQLMySQL.mqh>
//+------------------------------------------------------------------+
#define PATH_VAR_BASE  TerminalInfoString(TERMINAL_PATH)+"\\MQL5\\Scripts\\MyConnection.ini"
//+------------------------------------------------------------------+
class SQL_CONF
  {
private:
   string INI;
   
   string host;               // -> z.B.: 127.0.0.1
   string user;               // -> z.B.: root
   string password;           // -> z.B.: 
   string database;           // -> z.B.: test
   string socket;             // -> z.B.: 0
   string act_symbol;          // -> z.B.: 0
   
   int port;                  // -> z.B.: 3306
   int clientFlag;            // -> z.B.: 
   int identifier;            // -> z.B.: 
   
   string path_to_ini;
   
   const string initiator;    // how has created an instance
   
   string table;              // tablename wich has been submitted least
   //const string INI_Path = PATH_VAR;
   //string sql_version = "asd";

public:
                     SQL_CONF(const string&,);
                    ~SQL_CONF();
                    int initialize_db(string, string, string, string, string, string, int, int, int , const string);
                    int connect_db();                             
                    int disconnect_db();                          
                    int write_db(const MqlRates& [], const int&, string);                   // write given point information to the database
                    int read_db(const MqlRates& [], datetime);                              // read specific point and all info at position {datetime}
                    int create_table(const string& ,string& , int);                         // 
                    int set_newData_flag(string, bool, string&);                            // table, flag            
  };
//+------------------------------------------------------------------+
//|KONSTRUCTOR                                                                  |
//+------------------------------------------------------------------+
SQL_CONF::SQL_CONF(const string& init): initiator(init)
  {
     Print("New SQL Configuration created by: ",initiator);
     return; 
  }
//+------------------------------------------------------------------+
//|DEKONSTRUCTOR                                                                  |
//+------------------------------------------------------------------+
SQL_CONF::~SQL_CONF()
  {
  }
//+------------------------------------------------------------------+
//|INITIALIZE_DB                                                                  |
//+------------------------------------------------------------------+
int SQL_CONF::initialize_db(string Host = "", string User = "", string Password = "", string Database = "", 
                              string Socket = "", string Act_Symbol = "", int Port = 0, int ClientFlag = 0, int Identifier = 0, const string path = "")
  {
  
      //--------INITIALIZE----------//
      host = Host;
      user = User;
      password = Password;
      database = Database;
      socket = Socket;
      act_symbol = Act_Symbol;
      port = Port;
      clientFlag = ClientFlag;
      identifier = Identifier;
      //--------SET INI PATH----------//
      INI = PATH_VAR_BASE + path;
      Print(INI); 
      //--------DATA VALIDATION----------//
      if(host == "") host = ReadIni(INI, "MYSQL", "Host");
      if(user == "") user = ReadIni(INI, "MYSQL", "User");
      if(password == "") password = ReadIni(INI, "MYSQL", "Password");
      if(database == "") database = ReadIni(INI, "MYSQL", "Database");
      if(port == 0) port = (int)StringToInteger(ReadIni(INI, "MYSQL", "Port"));
      if(socket == "") socket = ReadIni(INI, "MYSQL", "Socket");
      if(clientFlag == 0) clientFlag = CLIENT_MULTI_STATEMENTS; //(int)StringToInteger(ReadIni(INI, "MYSQL", "ClientFlag"));  

      Print("Initialized following parameters: \n");
      Print("Host: ", host);
      Print("User: ", user);
      
      if(password != "") { Print("Password: [JA]");} else { Print("Password: [NEIN]");};
      Print("Database: ", database);
      Print("Port: ", port);
      Print("Socket: ", socket);
      Print("ClientFlag: ", clientFlag);
      
      return 1;
  }
//+------------------------------------------------------------------+
//|CONNECT_TO_DB                                                     |
//+------------------------------------------------------------------+
int SQL_CONF::connect_db()
{
      identifier = MySqlConnect(host, user, password, database, port, socket, clientFlag);
 
      if (identifier == -1) { 
         Print ("Connection failed! Error: "+MySqlErrorDescription); 
      } 
      else { 
         Print ("Connected! DBID#",identifier);
         Print ("DB: ",database);
      }
      return 1;
}                            
//+------------------------------------------------------------------+
//|DISCONNET_FROM_DB                                                     |
//+------------------------------------------------------------------+
int SQL_CONF::disconnect_db()
{
      MySqlDisconnect(identifier);
      Print ("Database Disconnected");
      return 1;
}                      
//+------------------------------------------------------------------+
//|WRITE_DB                                                     |
//+------------------------------------------------------------------+
int SQL_CONF::write_db(const MqlRates& hist_data[],const int& nr_elements, string tab = "")
{
      string Query;
      string clearQuery = "TRUNCATE test.historydata;";
      
      if (tab != "") {
          table = tab;
      }
      else {
         Print ("No Table has been submitted. Use last submitted table" + table);
      }      
      
      //Clear Table
      MySqlExecute(identifier, clearQuery);
      
      // multi-insert
      
      Print("Start copying data to table...\n");
      for(int count = 0; count < nr_elements ; count++) {
         Print(TimeToString(hist_data[count].time)," - ",hist_data[count].close);
         
         Query = "INSERT INTO `" + table + "` (timestamp, close, high, low, open, volume) VALUES ('"+
                                                                                       (string)hist_data[count].time+"','"+
                                                                                       (string)hist_data[count].close+"','"+
                                                                                       (string)hist_data[count].high+"','"+
                                                                                       (string)hist_data[count].low+"','"+
                                                                                       (string)hist_data[count].open+"','"+
                                                                                       (string)hist_data[count].tick_volume+"');";
        if (MySqlExecute(identifier, Query))
           {
               //printf("i% % done",(count/nr_elements*100));
           }
        else
           {
            Print ("Error of multiple statements: ", MySqlErrorDescription);
            return -1;
           }
      }
      
      Print("Complete! Values added to table...");
      return 1;
}
//+------------------------------------------------------------------+
//|READ_DB                                                           |
//+------------------------------------------------------------------+
int SQL_CONF::read_db(const MqlRates& readback_data[], datetime readback_time)
{
      return 1;
}
//+------------------------------------------------------------------+
//|CREATE_TABLE                                                      |
//+------------------------------------------------------------------+
int SQL_CONF::create_table(const string& tab, string& Error, int tab_nr = 1) 
{
      table = tab;
      string Query; 
      
      switch (tab_nr)
      {
         case 1: 
            Query = "CREATE TABLE `" + table + "` ("+
                                       "id int NOT NULL AUTO_INCREMENT, " +
                                       "timestamp datetime, " +
                                       "close double, " + 
                                       "high double, " +
                                       "low double, " +
                                       "open double, " +
                                       "volume int," + 
                                       "PRIMARY KEY (ID))";
            break;
         case 2:
            Query = "CREATE TABLE `" + table + "` (flag int)";           
            break;
         case 99:
            break;
         default:
         break;
      }
      
      
      
      if (MySqlExecute(identifier, Query)) {
         Print ("Table " + table + " created.");
         return 1;
      }
      else {
         Print ("Error occured...");
         Print ("Error Nr.: ", MySqlErrorNumber);
         Print ("Error: ", MySqlErrorDescription);
         Error = string(MySqlErrorNumber);
         return -1;
      }
}
//+------------------------------------------------------------------+
//|Set Flag                                                      |
//+------------------------------------------------------------------+
int SQL_CONF::set_newData_flag(string flag_tab, bool flag, string& Error)
{
   string clearQuery = "TRUNCATE test.new_bar_flag;";
   string Query; 
   if (flag) {
      Query = "INSERT INTO `" + flag_tab + "` (flag) VALUES ("+(string)flag+");";
      
      if (MySqlExecute(identifier, clearQuery) && MySqlExecute(identifier, Query)) {
         Print ("Table " + table + " created.");
         return 1;
      }
      else {
         Print ("Flag Table Error occured...");
         Print ("Error Nr.: ", MySqlErrorNumber);
         Print ("Error: ", MySqlErrorDescription);
         Error = string(MySqlErrorNumber);
         return -1;
      }
   }
   else {}
   return 0;
}

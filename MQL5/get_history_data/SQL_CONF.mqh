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
#define PATH_VAR_BASE  ""
//+------------------------------------------------------------------+
class SQL_CONF
  {
private:
   string INI;
   
   string host;               // -> z.B.: root
   string user;               // -> z.B.: 
   string password;           // -> z.B.: 
   string database;           // -> z.B.: 
   string socket;             // -> z.B.: 
   string act_symbol;          // -> z.B.: 
   
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
                    int read_db(const MqlRates& [], datetime);          // read specific point and all info at position {datetime}
                    int create_table(const string& ,string& );                  
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
       
      //--------DATA VALIDATION----------//
      if(host == "") host = ReadIni(INI, "MYSQL", "Host");
      if(user == "") user = ReadIni(INI, "MYSQL", "User");
      if(password == "") password = ReadIni(INI, "MYSQL", "Password");
      if(database == "") database = ReadIni(INI, "MYSQL", "Database");
      if(port == 0) port = (int)StringToInteger(ReadIni(INI, "MYSQL", "Port"));
      if(socket == "") socket = ReadIni(INI, "MYSQL", "Socket");
      if(clientFlag == 0) clientFlag = CLIENT_MULTI_STATEMENTS; //(int)StringToInteger(ReadIni(INI, "MYSQL", "ClientFlag"));  

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
      
      if (tab != "") {
          table = tab;
      }
      else {
         Print ("No Table has been submitted. Use last submitted table" + table);
      }      
      
      // multi-insert
      
      Print("Start copying data to table...\n");
      for(int count = 0; count < nr_elements-1 ; count++) {
         Print(TimeToString(hist_data[count].time)," - ",hist_data[count].close);
         
        Query = "INSERT INTO `" + table + "` (id, code, start_date, closevalue, openvalue, highvalue, lowvalue) VALUES ("+
                                                                                       (string)count+",\'EURUSD\',\'"+
                                                                                       (string)hist_data[count].time+"\',\'"+
                                                                                       (string)hist_data[count].close+"\',\'"+
                                                                                       (string)hist_data[count].open+"\',\'"+
                                                                                       (string)hist_data[count].high+"\',\'"+
                                                                                       (string)hist_data[count].low+"\');";
        
        if (MySqlExecute(identifier, Query))
           {
               Print (count/nr_elements*100," % done");
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
int SQL_CONF::create_table(const string& tab, string& Error) 
{
      table = tab;
      string Query; 
      
      Query = "CREATE TABLE `" + table + "` (id int, code varchar(50), start_date datetime, closevalue double, openvalue double, highvalue double, lowvalue double)";
      
      if (MySqlExecute(identifier, Query)) {
         Print ("Table " + table + " created.");
         return 1;
      }
      else {
         Print ("Error occured...");
         Print ("Error Nr.: ", MySqlErrorNumber);
         Print ("Error: ", MySqlErrorDescription);
         return -1;
      }
}
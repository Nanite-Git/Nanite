//+------------------------------------------------------------------+
//|                                               Demo_FileWrite.mq4 |
//|                        Copyright 2014, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//--- show the window of input parameters when launching the script
#property script_show_inputs
//--- parameters for receiving data from the terminal
input string             InpSymbolName="EURUSD";      // Сurrency pair
input ENUM_TIMEFRAMES    InpSymbolPeriod=PERIOD_H1;   // Time frame
input int                InpFastEMAPeriod=12;         // Fast EMA period
input int                InpSlowEMAPeriod=26;         // Slow EMA period
input int                InpSignalPeriod=9;           // Difference averaging period
input ENUM_APPLIED_PRICE InpAppliedPrice=PRICE_CLOSE; // Price type
//--- parameters for writing data to file
input string             InpFileName="MACD.csv";      // File name
input string             InpDirectoryName="Data";     // Folder name
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
  
   datetime t_start = D'1990.01.01 00:00:00', t_stop = D'2017.07.06 00:00:00';
   int r_ch;          // returnvalue counter history-values
   MqlRates hist_data[];
   const string name = "nanite-holg";
 
   r_ch = CopyRates(Symbol(),PERIOD_D1,t_start,t_stop,hist_data);      // get histor values in 'hist_data'
     
   //Query = (string)count+(string)hist_data[count].time+"\',\'"+(string)hist_data[count].close+"\',\'"+(string)hist_data[count].open+"\',\'"+(string)hist_data[count].high+"\',\'"+(string)hist_data[count].low+"\');";

   
 
   int file_handle=FileOpen(InpDirectoryName+"//"+InpFileName,FILE_WRITE|FILE_CSV|FILE_BIN);
   if(file_handle!=INVALID_HANDLE)
     {
      PrintFormat("%s file is available for writing",InpFileName);
      PrintFormat("File path: %s\\Files\\",TerminalInfoString(TERMINAL_DATA_PATH));
      //--- first, write the number of signals

      //--- write the time and values of signals to the file
      for(int count = 0; count < r_ch-1 ; count++) {
         FileWrite(file_handle,(float)hist_data[count].open);
      }
      //--- close the file
      FileClose(file_handle);
      PrintFormat("Data is written, %s file is closed",InpFileName);
     }
   else
      PrintFormat("Failed to open %s file, Error code = %d",InpFileName,GetLastError());
  }
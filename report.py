
from PySide6.QtPrintSupport import QPrintPreviewDialog, QPrinter, QPrintDialog
from PySide6.QtGui import QTextDocument, QTextCursor, QTextCharFormat, QFont
import pandas as pd

class ReportPrinter:
    def __init__(self, parent=None):
        self.parent = parent
    
    def print_data(self, data, title="Report"):
        """Печать данных (DataFrame или QTableWidget)"""
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        print_dialog = QPrintDialog(printer, self.parent)
        
        if print_dialog.exec() == QPrintDialog.DialogCode.Accepted:
            doc = QTextDocument()
            cursor = QTextCursor(doc)
            
            cursor.insertText(f"{title}\n", self.get_title_format())
            cursor.insertBlock()
            
            if isinstance(data, pd.DataFrame):
                self._print_dataframe(cursor, data)
            else:
                raise ValueError("Unsupported data type for printing")
            
            doc.print_(printer)
    
    def print_preview(self, data, title="Report"):
        """Предпросмотр печати"""
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        preview = QPrintPreviewDialog(printer, self.parent)
        preview.setWindowTitle("Print Preview")
        
        def print_preview(printer):
            doc = QTextDocument()
            cursor = QTextCursor(doc)
            
            cursor.insertText(f"{title}\n", self.get_title_format())
            cursor.insertBlock()
            
            if isinstance(data, pd.DataFrame):
                self._print_dataframe(cursor, data)
            
            doc.print_(printer)
        
        preview.paintRequested.connect(print_preview)
        preview.exec()
    
    def _print_dataframe(self, cursor, df):
        """Форматирование DataFrame для печати"""
        html = df.to_html(index=False, border=1)
        cursor.insertHtml(html)
        cursor.insertBlock()
    
    
    def get_title_format(self):
        """Формат заголовка"""
        format = QTextCharFormat()
        format.setFontPointSize(14)
        format.setFontWeight(QFont.Weight.Bold)
        format.setVerticalAlignment(QTextCharFormat.AlignMiddle)
        return format
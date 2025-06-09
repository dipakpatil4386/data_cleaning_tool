import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import json
import os
from datetime import datetime
import threading
import queue
import re
from typing import Dict, List, Tuple, Any

class DataCleaningAgent:
    def __init__(self):
        self.cleaning_memory = {}
        self.current_data = None
        self.original_data = None
        self.cleaning_history = []
        self.data_profile = None
        
    def load_cleaning_memory(self):
        """Load previous cleaning patterns from memory file"""
        try:
            if os.path.exists('cleaning_memory.json'):
                with open('cleaning_memory.json', 'r') as f:
                    self.cleaning_memory = json.load(f)
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def save_cleaning_memory(self):
        """Save cleaning patterns to memory file"""
        try:
            with open('cleaning_memory.json', 'w') as f:
                json.dump(self.cleaning_memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def analyze_data(self, df):
        """Analyze data and detect issues"""
        issues = {
            'null_values': {},
            'outliers': {},
            'formatting_issues': {},
            'duplicates': 0,
            'data_types': {}
        }
        
        # Check for null values
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues['null_values'][col] = {
                    'count': int(null_count),
                    'percentage': float(null_count / len(df) * 100)
                }
        
        # Check for outliers (numerical columns)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                issues['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': float(len(outliers) / len(df) * 100)
                }
        
        # Check for formatting issues
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Check for mixed case issues
                unique_values = df[col].dropna().astype(str)
                if len(unique_values) > 0:
                    case_variations = len(set(unique_values.str.lower())) != len(set(unique_values))
                    if case_variations:
                        issues['formatting_issues'][col] = 'mixed_case'
                    
                    # Check for whitespace issues
                    has_whitespace = any(val != val.strip() for val in unique_values if isinstance(val, str))
                    if has_whitespace:
                        if col not in issues['formatting_issues']:
                            issues['formatting_issues'][col] = []
                        elif not isinstance(issues['formatting_issues'][col], list):
                            issues['formatting_issues'][col] = [issues['formatting_issues'][col]]
                        issues['formatting_issues'][col].append('whitespace')
        
        # Check for duplicates
        issues['duplicates'] = len(df[df.duplicated()])
        
        # Data type recommendations
        for col in df.columns:
            current_type = str(df[col].dtype)
            issues['data_types'][col] = current_type
        
        return issues
    
    def suggest_cleaning_actions(self, issues):
        """Suggest cleaning actions based on detected issues"""
        suggestions = []
        
        # Null value suggestions
        for col, info in issues['null_values'].items():
            if info['percentage'] < 5:
                suggestions.append({
                    'action': 'drop_nulls',
                    'column': col,
                    'description': f"Drop {info['count']} null values in '{col}' ({info['percentage']:.1f}%)",
                    'confidence': 0.8
                })
            elif info['percentage'] < 30:
                suggestions.append({
                    'action': 'fill_nulls',
                    'column': col,
                    'description': f"Fill {info['count']} null values in '{col}' with median/mode",
                    'confidence': 0.6
                })
        
        # Outlier suggestions
        for col, info in issues['outliers'].items():
            if info['percentage'] < 2:
                suggestions.append({
                    'action': 'remove_outliers',
                    'column': col,
                    'description': f"Remove {info['count']} outliers in '{col}' ({info['percentage']:.1f}%)",
                    'confidence': 0.7
                })
        
        # Formatting suggestions
        for col, issue_type in issues['formatting_issues'].items():
            if issue_type == 'mixed_case' or 'mixed_case' in issue_type:
                suggestions.append({
                    'action': 'standardize_case',
                    'column': col,
                    'description': f"Standardize case in '{col}'",
                    'confidence': 0.9
                })
            if 'whitespace' in str(issue_type):
                suggestions.append({
                    'action': 'strip_whitespace',
                    'column': col,
                    'description': f"Remove leading/trailing whitespace in '{col}'",
                    'confidence': 0.95
                })
        
        # Duplicate suggestions
        if issues['duplicates'] > 0:
            suggestions.append({
                'action': 'remove_duplicates',
                'column': 'all',
                'description': f"Remove {issues['duplicates']} duplicate rows",
                'confidence': 0.85
            })
        
        return suggestions
    
    def apply_cleaning_action(self, df, action):
        """Apply a specific cleaning action"""
        df_copy = df.copy()
        
        try:
            if action['action'] == 'drop_nulls':
                df_copy = df_copy.dropna(subset=[action['column']])
            
            elif action['action'] == 'fill_nulls':
                col = action['column']
                if df_copy[col].dtype in ['int64', 'float64']:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                else:
                    df_copy[col].fillna(df_copy[col].mode().iloc[0] if not df_copy[col].mode().empty else 'Unknown', inplace=True)
            
            elif action['action'] == 'remove_outliers':
                col = action['column']
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
            
            elif action['action'] == 'standardize_case':
                col = action['column']
                df_copy[col] = df_copy[col].astype(str).str.lower()
            
            elif action['action'] == 'strip_whitespace':
                col = action['column']
                df_copy[col] = df_copy[col].astype(str).str.strip()
            
            elif action['action'] == 'remove_duplicates':
                df_copy = df_copy.drop_duplicates()
            
            return df_copy, True, ""
            
        except Exception as e:
            return df, False, str(e)


class DataCleaningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Data Cleaning Agent")
        self.root.geometry("1200x800")
        
        self.agent = DataCleaningAgent()
        self.agent.load_cleaning_memory()
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data Loading Tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Loading")
        self.setup_data_tab()
        
        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.setup_analysis_tab()
        
        # Cleaning Tab
        self.cleaning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cleaning_frame, text="Cleaning Actions")
        self.setup_cleaning_tab()
        
        # Results Tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()
    
    def setup_data_tab(self):
        # File loading section
        load_frame = ttk.LabelFrame(self.data_frame, text="Load Data")
        load_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(load_frame, text="Load CSV File", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Load Excel File", command=self.load_excel).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(load_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=20)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create Treeview for data display
        self.data_tree = ttk.Treeview(preview_frame)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.configure(xscrollcommand=h_scrollbar.set)
    
    def setup_analysis_tab(self):
        # Analysis controls
        control_frame = ttk.Frame(self.analysis_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Analyze Data", command=self.analyze_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Analysis results
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, height=30)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_cleaning_tab(self):
        # Suggested actions frame
        suggestions_frame = ttk.LabelFrame(self.cleaning_frame, text="Suggested Cleaning Actions")
        suggestions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Actions listbox with scrollbar
        listbox_frame = ttk.Frame(suggestions_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.actions_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE)
        self.actions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        actions_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.actions_listbox.yview)
        actions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.actions_listbox.configure(yscrollcommand=actions_scrollbar.set)
        
        # Action buttons
        button_frame = ttk.Frame(suggestions_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Apply Selected", command=self.apply_selected_actions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply All High Confidence", command=self.apply_high_confidence).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Undo Last Action", command=self.undo_last_action).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(suggestions_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_results_tab(self):
        # Results summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Cleaning Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=10)
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Export options
        export_frame = ttk.LabelFrame(self.results_frame, text="Export Cleaned Data")
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(export_frame, text="Export to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export to Excel", command=self.export_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Generate Cleaning Report", command=self.export_report).pack(side=tk.LEFT, padx=5)
        
        # Cleaned data preview
        cleaned_preview_frame = ttk.LabelFrame(self.results_frame, text="Cleaned Data Preview")
        cleaned_preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.cleaned_tree = ttk.Treeview(cleaned_preview_frame)
        self.cleaned_tree.pack(fill=tk.BOTH, expand=True)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.agent.current_data = pd.read_csv(file_path)
                self.agent.original_data = self.agent.current_data.copy()
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.update_data_preview()
                messagebox.showinfo("Success", "CSV file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def load_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("Excel files", "*.xls")])
        if file_path:
            try:
                self.agent.current_data = pd.read_excel(file_path)
                self.agent.original_data = self.agent.current_data.copy()
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.update_data_preview()
                messagebox.showinfo("Success", "Excel file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Excel: {str(e)}")
    
    def update_data_preview(self):
        if self.agent.current_data is not None:
            # Clear existing data
            self.data_tree.delete(*self.data_tree.get_children())
            
            # Configure columns
            columns = list(self.agent.current_data.columns)
            self.data_tree['columns'] = columns
            self.data_tree['show'] = 'headings'
            
            # Configure column headers
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            # Insert data (first 100 rows for performance)
            for index, row in self.agent.current_data.head(100).iterrows():
                self.data_tree.insert('', 'end', values=list(row))
    
    def analyze_data(self):
        if self.agent.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        self.progress.start()
        
        def analyze_thread():
            try:
                issues = self.agent.analyze_data(self.agent.current_data)
                suggestions = self.agent.suggest_cleaning_actions(issues)
                
                # Update UI in main thread
                self.root.after(0, self.update_analysis_results, issues, suggestions)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def update_analysis_results(self, issues, suggestions):
        # Clear previous results
        self.analysis_text.delete(1.0, tk.END)
        self.actions_listbox.delete(0, tk.END)
        
        # Display analysis results
        analysis_report = "=== DATA QUALITY ANALYSIS ===\n\n"
        
        # Data shape
        analysis_report += f"Dataset Shape: {self.agent.current_data.shape[0]} rows, {self.agent.current_data.shape[1]} columns\n\n"
        
        # Null values
        if issues['null_values']:
            analysis_report += "NULL VALUES DETECTED:\n"
            for col, info in issues['null_values'].items():
                analysis_report += f"  • {col}: {info['count']} nulls ({info['percentage']:.1f}%)\n"
            analysis_report += "\n"
        
        # Outliers
        if issues['outliers']:
            analysis_report += "OUTLIERS DETECTED:\n"
            for col, info in issues['outliers'].items():
                analysis_report += f"  • {col}: {info['count']} outliers ({info['percentage']:.1f}%)\n"
            analysis_report += "\n"
        
        # Formatting issues
        if issues['formatting_issues']:
            analysis_report += "FORMATTING ISSUES:\n"
            for col, issue_type in issues['formatting_issues'].items():
                analysis_report += f"  • {col}: {issue_type}\n"
            analysis_report += "\n"
        
        # Duplicates
        if issues['duplicates'] > 0:
            analysis_report += f"DUPLICATE ROWS: {issues['duplicates']}\n\n"
        
        self.analysis_text.insert(1.0, analysis_report)
        
        # Store suggestions for cleaning actions
        self.current_suggestions = suggestions
        
        # Populate suggestions listbox
        for i, suggestion in enumerate(suggestions):
            confidence_str = f"({suggestion['confidence']:.0%} confidence)"
            self.actions_listbox.insert(tk.END, f"{suggestion['description']} {confidence_str}")
    
    def apply_selected_actions(self):
        if not hasattr(self, 'current_suggestions'):
            messagebox.showwarning("Warning", "Please analyze data first!")
            return
        
        selected_indices = self.actions_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select actions to apply!")
            return
        
        self.apply_actions([self.current_suggestions[i] for i in selected_indices])
    
    def apply_high_confidence(self):
        if not hasattr(self, 'current_suggestions'):
            messagebox.showwarning("Warning", "Please analyze data first!")
            return
        
        high_confidence_actions = [action for action in self.current_suggestions if action['confidence'] >= 0.8]
        if not high_confidence_actions:
            messagebox.showinfo("Info", "No high confidence actions available!")
            return
        
        self.apply_actions(high_confidence_actions)
    
    def apply_actions(self, actions):
        self.progress.start()
        
        def apply_thread():
            try:
                applied_actions = []
                for action in actions:
                    cleaned_data, success, error = self.agent.apply_cleaning_action(self.agent.current_data, action)
                    if success:
                        self.agent.current_data = cleaned_data
                        applied_actions.append(action)
                        self.agent.cleaning_history.append({
                            'action': action,
                            'timestamp': datetime.now().isoformat(),
                            'rows_before': len(self.agent.current_data),
                            'rows_after': len(cleaned_data)
                        })
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to apply {action['description']}: {error}"))
                
                # Update UI
                self.root.after(0, self.update_after_cleaning, applied_actions)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Cleaning failed: {str(e)}"))
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=apply_thread, daemon=True).start()
    
    def update_after_cleaning(self, applied_actions):
        if applied_actions:
            # Update data preview
            self.update_data_preview()
            self.update_cleaned_preview()
            
            # Update summary
            summary = f"Applied {len(applied_actions)} cleaning actions:\n"
            for action in applied_actions:
                summary += f"  • {action['description']}\n"
            
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary)
            
            # Switch to results tab
            self.notebook.select(3)
            
            messagebox.showinfo("Success", f"Successfully applied {len(applied_actions)} cleaning actions!")
    
    def update_cleaned_preview(self):
        if self.agent.current_data is not None:
            # Clear existing data
            self.cleaned_tree.delete(*self.cleaned_tree.get_children())
            
            # Configure columns
            columns = list(self.agent.current_data.columns)
            self.cleaned_tree['columns'] = columns
            self.cleaned_tree['show'] = 'headings'
            
            # Configure column headers
            for col in columns:
                self.cleaned_tree.heading(col, text=col)
                self.cleaned_tree.column(col, width=100)
            
            # Insert data (first 100 rows for performance)
            for index, row in self.agent.current_data.head(100).iterrows():
                self.cleaned_tree.insert('', 'end', values=list(row))
    
    def undo_last_action(self):
        if not self.agent.cleaning_history:
            messagebox.showinfo("Info", "No actions to undo!")
            return
        
        # For simplicity, reload original data and reapply all but last action
        # In a production system, you'd want more sophisticated undo functionality
        self.agent.current_data = self.agent.original_data.copy()
        self.agent.cleaning_history.pop()
        
        if self.agent.cleaning_history:
            actions_to_reapply = [entry['action'] for entry in self.agent.cleaning_history]
            self.apply_actions(actions_to_reapply)
        else:
            self.update_data_preview()
            messagebox.showinfo("Success", "Undid last action!")
    
    def export_csv(self):
        if self.agent.current_data is None:
            messagebox.showwarning("Warning", "No data to export!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.agent.current_data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def export_excel(self):
        if self.agent.current_data is None:
            messagebox.showwarning("Warning", "No data to export!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                self.agent.current_data.to_excel(file_path, index=False)
                messagebox.showinfo("Success", "Data exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def export_report(self):
        if not self.agent.cleaning_history:
            messagebox.showwarning("Warning", "No cleaning actions performed yet!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                report = "=== AUTONOMOUS DATA CLEANING REPORT ===\n\n"
                report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                if self.agent.original_data is not None:
                    report += f"Original data shape: {self.agent.original_data.shape}\n"
                if self.agent.current_data is not None:
                    report += f"Cleaned data shape: {self.agent.current_data.shape}\n\n"
                
                report += "CLEANING ACTIONS APPLIED:\n"
                for i, entry in enumerate(self.agent.cleaning_history, 1):
                    action = entry['action']
                    report += f"{i}. {action['description']}\n"
                    report += f"   Confidence: {action['confidence']:.0%}\n"
                    report += f"   Applied: {entry['timestamp']}\n\n"
                
                with open(file_path, 'w') as f:
                    f.write(report)
                
                messagebox.showinfo("Success", "Cleaning report exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Report export failed: {str(e)}")
    
    def generate_report(self):
        if self.agent.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        messagebox.showinfo("Info", "Pandas Profiling report generation would open in browser.\nThis is a placeholder for the full implementation.")


def main():
    root = tk.Tk()
    app = DataCleaningGUI(root)
    
    # Handle window close
    def on_closing():
        app.agent.save_cleaning_memory()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

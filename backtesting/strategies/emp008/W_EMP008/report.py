import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import seaborn as sns


class ReportGenerator:
    def __init__(self, result: Dict[str, Any], runner = None, date1: str = None, date2: str = None):
        self.result = result
        self.runner = runner
        self.date1 = date1
        self.date2 = date2
        self.fig_width = 11.69
        self.fig_height = 8.27
        
    def generate_pdf_report(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_optimization_report_{timestamp}.pdf"
            
        with PdfPages(filename) as pdf:
            self._create_summary_page(pdf)
            self._create_weights_page(pdf)
            if self.runner is not None:
                self._create_data_overview_page(pdf)
                self._create_matrix_details_page(pdf)
            
        print(f"Report saved: {filename}")
        return filename
    
    def _create_summary_page(self, pdf):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
        fig.suptitle('Portfolio Optimization Summary', fontsize=16, fontweight='bold')
        
        date1_str = self.date1 if self.date1 else (self.runner.last_date1 if self.runner else 'N/A')
        date2_str = self.date2 if self.date2 else (self.runner.last_date2 if self.runner else 'N/A')
        
        metrics_data = [
            ['Objective Value', f"{self.result['objective_value']:.6f}"],
            ['Tracking Error', f"{self.result['tracking_error']:.6f}"],
            ['Date1 (Alpha)', date1_str],
            ['Date2 (Factors)', date2_str]
        ]
        
        ax1.axis('tight')
        ax1.axis('off')
        table = ax1.table(cellText=metrics_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2.5)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
        
        active_weights = self.result['active_weights']
        significant = active_weights[abs(active_weights) > 0.001]
        
        ax2.text(0.5, 0.9, 'Position Statistics', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        stats_text = f"""Total Positions: {len(active_weights)}
                         Significant (>0.1%): {len(significant)}
                         Max Long: {active_weights.max():.4f}
                         Max Short: {active_weights.min():.4f}
                         Gross Exposure: {abs(active_weights).sum():.4f}"""
        
        ax2.text(0.05, 0.7, stats_text, ha='left', va='top', fontsize=11, 
                fontfamily='monospace')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        top_positions = significant.abs().nlargest(10)
        if len(top_positions) > 0:
            colors = ['green' if significant[ticker] > 0 else 'red' for ticker in top_positions.index]
            bars = ax3.barh(range(len(top_positions)), 
                           [significant[ticker] for ticker in top_positions.index],
                           color=colors, alpha=0.7)
            ax3.set_yticks(range(len(top_positions)))
            ax3.set_yticklabels([str(ticker) for ticker in top_positions.index])
            ax3.set_xlabel('Active Weight')
            ax3.set_title('Top 10 Active Positions', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            for i, (ticker, weight) in enumerate(zip(top_positions.index, 
                                                   [significant[ticker] for ticker in top_positions.index])):
                ax3.text(weight + (0.001 if weight > 0 else -0.001), i, f'{weight:.3f}', 
                        va='center', ha='left' if weight > 0 else 'right', fontsize=9)
        
        significant_sorted = significant.abs().sort_values(ascending=False)
        top_20 = significant_sorted.head(20)
        
        table_data = []
        for ticker in top_20.index:
            weight = significant[ticker]
            table_data.append([str(ticker), f"{weight:.4f}"])
        
        ax4.axis('tight')
        ax4.axis('off')
        if len(table_data) > 0:
            table2 = ax4.table(cellText=table_data,
                              colLabels=['Ticker', 'Active Weight'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.5, 0.5])
            table2.auto_set_font_size(False)
            table2.set_fontsize(8)
            table2.scale(1, 1.2)
            
            for (i, j), cell in table2.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#E6E6FA')
                elif i > 0 and j == 1:
                    weight_val = float(table_data[i-1][1])
                    if weight_val > 0:
                        cell.set_facecolor('#E8F5E8')
                    else:
                        cell.set_facecolor('#FFE8E8')
        
        ax4.set_title('Top 20 Active Positions', fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_weights_page(self, pdf):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
        fig.suptitle('Portfolio Weights Analysis', fontsize=16, fontweight='bold')
        
        active_weights = self.result['active_weights']
        optimal_weights = self.result['optimal_weights']
        
        ax1.hist(active_weights.values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Active Weight')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Active Weights Distribution')
        ax1.grid(alpha=0.3)
        
        ax2.hist(optimal_weights.values, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
        ax2.set_xlabel('Optimal Weight')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Optimal Weights Distribution')
        ax2.grid(alpha=0.3)
        
        scatter_sample = min(len(active_weights), 1000)
        sample_idx = np.random.choice(len(active_weights), scatter_sample, replace=False)
        ax3.scatter(optimal_weights.iloc[sample_idx], active_weights.iloc[sample_idx], 
                   alpha=0.6, s=10)
        ax3.set_xlabel('Optimal Weight')
        ax3.set_ylabel('Active Weight')
        ax3.set_title('Optimal vs Active Weights')
        ax3.grid(alpha=0.3)
        
        significant = active_weights[abs(active_weights) > 0.001]
        long_positions = significant[significant > 0]
        short_positions = significant[significant < 0]
        
        summary_text = f"""Position Summary:
        
                            Long Positions: {len(long_positions)}
                            Short Positions: {len(short_positions)}
                            Zero Positions: {len(active_weights) - len(significant)}

                            Long Exposure: {long_positions.sum():.4f}
                            Short Exposure: {short_positions.sum():.4f}
                            Net Exposure: {active_weights.sum():.4f}
                            Gross Exposure: {abs(active_weights).sum():.4f}

                            Weight Statistics:
                            Mean: {active_weights.mean():.6f}
                            Std: {active_weights.std():.6f}
                            Min: {active_weights.min():.6f}
                            Max: {active_weights.max():.6f}"""
        
        ax4.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=10, 
                fontfamily='monospace', transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_data_overview_page(self, pdf):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
        fig.suptitle('Input Data Overview', fontsize=16, fontweight='bold')
        
        factors = self.runner.current_factors
        factor_cov = self.runner.current_covariance
        residual_cov = self.runner.current_residual_cov
        alpha = self.runner.current_alpha
        
        ax1.hist(alpha.values, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax1.set_xlabel('Expected Alpha')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Alpha Distribution (Mean: {alpha.mean():.4f})')
        ax1.grid(alpha=0.3)
        
        im2 = ax2.imshow(factor_cov.values, cmap='RdBu_r', aspect='auto')
        ax2.set_title('Factor Covariance Matrix')
        ax2.set_xlabel('Factors')
        ax2.set_ylabel('Factors')
        ax2.set_xticks(range(len(factor_cov.columns)))
        ax2.set_xticklabels(factor_cov.columns, rotation=45, ha='right')
        ax2.set_yticks(range(len(factor_cov.index)))
        ax2.set_yticklabels(factor_cov.index)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        residual_diag = np.diag(residual_cov.values)
        ax3.hist(residual_diag, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Residual Variance')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Residual Risk Distribution (Mean: {residual_diag.mean():.6f})')
        ax3.grid(alpha=0.3)
        
        data_summary = f"""Data Summary:

                            Factors: {factors.shape[0]} stocks × {factors.shape[1]} factors
                            Factor Covariance: {factor_cov.shape[0]} × {factor_cov.shape[1]}
                            Residual Covariance: {residual_cov.shape[0]} × {residual_cov.shape[1]}
                            Alpha: {len(alpha)} stocks

                            Factor Statistics:
                            Mean exposure: {factors.mean().mean():.4f}
                            Std exposure: {factors.std().mean():.4f}

                            Alpha Statistics:
                            Mean: {alpha.mean():.6f}
                            Std: {alpha.std():.6f}
                            Range: [{alpha.min():.4f}, {alpha.max():.4f}]

                            Risk Statistics:
                            Avg Factor Vol: {np.sqrt(np.diag(factor_cov.values)).mean():.4f}
                            Avg Residual Vol: {np.sqrt(residual_diag).mean():.4f}"""
        
        ax4.text(0.05, 0.95, data_summary, ha='left', va='top', fontsize=9, 
                fontfamily='monospace', transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_matrix_details_page(self, pdf):
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        fig.suptitle('Matrix Details (Excel Comparison)', fontsize=16, fontweight='bold')
        
        factors = self.runner.current_factors
        factor_cov = self.runner.current_covariance
        alpha = self.runner.current_alpha
        
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('tight')
        ax1.axis('off')
        ax1.set_title('Factor Covariance Matrix', fontweight='bold', pad=10)
        
        factor_cov_data = []
        for i, row_name in enumerate(factor_cov.index):
            row_data = [row_name]
            for j, col_name in enumerate(factor_cov.columns):
                row_data.append(f"{factor_cov.iloc[i, j]:.6f}")
            factor_cov_data.append(row_data)
        
        col_labels = [''] + list(factor_cov.columns)
        table1 = ax1.table(cellText=factor_cov_data,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(8)
        table1.scale(1, 1.5)
        
        for (i, j), cell in table1.get_celld().items():
            if i == 0 or j == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('tight')
        ax2.axis('off')
        ax2.set_title('Factor Loadings (First 10 Stocks)', fontweight='bold', pad=10)
        
        factors_sample = factors.head(10)
        factors_data = []
        for i, ticker in enumerate(factors_sample.index):
            row_data = [str(ticker)]
            for j, factor_name in enumerate(factors_sample.columns):
                row_data.append(f"{factors_sample.iloc[i, j]:.4f}")
            factors_data.append(row_data)
        
        col_labels2 = ['Ticker'] + list(factors_sample.columns)
        table2 = ax2.table(cellText=factors_data,
                          colLabels=col_labels2,
                          cellLoc='center',
                          loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(1, 1.5)
        
        for (i, j), cell in table2.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
        
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        ax3.set_title('Expected Alpha (First 20 Stocks)', fontweight='bold', pad=10)
        
        alpha_sample = alpha.head(20)
        alpha_data = []
        row_data = []
        for i, (ticker, alpha_val) in enumerate(alpha_sample.items()):
            if i % 10 == 0 and i > 0:
                alpha_data.append(row_data)
                row_data = []
            row_data.extend([str(ticker), f"{alpha_val:.6f}"])
        
        if row_data:
            while len(row_data) < 20:
                row_data.extend(['', ''])
            alpha_data.append(row_data)
        
        col_labels3 = []
        for i in range(10):
            col_labels3.extend([f'Ticker{i+1}', f'Alpha{i+1}'])
        
        table3 = ax3.table(cellText=alpha_data,
                          colLabels=col_labels3,
                          cellLoc='center',
                          loc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(7)
        table3.scale(1, 2)
        
        for (i, j), cell in table3.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
            elif i > 0 and j % 2 == 0:
                cell.set_facecolor('#F0F8FF')
        
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        ax4.set_title('Key Matrix Properties', fontweight='bold', pad=10)
        
        full_cov = self.runner.current_full_covariance
        eigenvals = np.linalg.eigvals(full_cov)
        
        matrix_props = f"""Matrix Properties for Excel Verification:

                            Factor Covariance Matrix ({factor_cov.shape[0]}×{factor_cov.shape[1]}):
                            Determinant: {np.linalg.det(factor_cov.values):.2e}
                            Condition Number: {np.linalg.cond(factor_cov.values):.2e}
                            Trace: {np.trace(factor_cov.values):.6f}

                            Full Covariance Matrix ({full_cov.shape[0]}×{full_cov.shape[1]}):
                            Condition Number: {np.linalg.cond(full_cov):.2e}
                            Min Eigenvalue: {eigenvals.min():.2e}
                            Max Eigenvalue: {eigenvals.max():.2e}
                            Negative Eigenvalues: {(eigenvals < 0).sum()}

                            Factor Loadings:
                            Mean Absolute Exposure: {factors.abs().mean().mean():.4f}
                            Max Absolute Exposure: {factors.abs().max().max():.4f}"""
        
        ax4.text(0.05, 0.95, matrix_props, ha='left', va='top', fontsize=9, 
                fontfamily='monospace', transform=ax4.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

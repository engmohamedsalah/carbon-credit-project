from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import qrcode
from io import BytesIO
import os
from datetime import datetime
import sqlite3
import tempfile


class ReportService:
    def __init__(self, db_path):
        self.db_path = db_path

    def generate_verification_certificate(self, verification_id, project_id):
        """Generate a professional verification certificate PDF"""
        
        # Get verification and project data
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get verification details
            cursor = conn.execute("""
                SELECT v.*, p.name as project_name, p.location_name, p.project_type, 
                       p.description, u.full_name as developer_name, u.email as developer_email
                FROM verifications v
                JOIN projects p ON v.project_id = p.id
                JOIN users u ON p.user_id = u.id
                WHERE v.id = ? AND v.project_id = ?
            """, (verification_id, project_id))
            
            data = cursor.fetchone()
            if not data:
                raise ValueError("Verification not found")
            
            verification = dict(data)

        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        # Container for the 'Flowable' objects
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            textColor=colors.darkgreen,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=TA_CENTER
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )

        # Title
        story.append(Paragraph("CARBON CREDIT VERIFICATION CERTIFICATE", title_style))
        story.append(Spacer(1, 20))
        
        # Certificate ID and status
        cert_info = f"Certificate ID: CC-{verification['id']:06d}"
        if verification['certificate_id']:
            cert_info += f" | Blockchain ID: {verification['certificate_id'][:16]}..."
        
        story.append(Paragraph(cert_info, subtitle_style))
        story.append(Paragraph(f"Status: <b>{verification['status'].upper()}</b>", subtitle_style))
        story.append(Spacer(1, 30))

        # Project Information
        story.append(Paragraph("PROJECT INFORMATION", header_style))
        
        project_data = [
            ['Project Name:', verification['project_name']],
            ['Location:', verification['location_name']],
            ['Project Type:', verification['project_type']],
            ['Developer:', verification['developer_name']],
            ['Contact:', verification['developer_email']],
        ]
        
        project_table = Table(project_data, colWidths=[2*inch, 4*inch])
        project_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(project_table)
        story.append(Spacer(1, 30))

        # Verification Details
        story.append(Paragraph("VERIFICATION RESULTS", header_style))
        
        verification_data = [
            ['Verification Date:', datetime.fromisoformat(verification['created_at']).strftime('%B %d, %Y')],
            ['Carbon Impact:', f"{verification['carbon_impact']:.2f} tonnes CO₂" if verification['carbon_impact'] else 'N/A'],
            ['AI Confidence:', f"{verification['ai_confidence']*100:.1f}%" if verification['ai_confidence'] else 'N/A'],
            ['Human Verified:', 'Yes' if verification['human_verified'] else 'No'],
            ['Blockchain Certified:', 'Yes' if verification['blockchain_certified'] else 'No'],
        ]
        
        verification_table = Table(verification_data, colWidths=[2*inch, 4*inch])
        verification_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(verification_table)
        story.append(Spacer(1, 30))

        # QR Code for verification
        qr_data = f"https://carbon-credits.app/verify/{verification_id}?cert={verification.get('certificate_id', 'pending')}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_buffer = BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)
        
        # Save QR code to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_qr:
            qr_img.save(temp_qr.name, format='PNG')
            temp_qr_path = temp_qr.name

        try:
            # Add QR code to PDF
            story.append(Paragraph("CERTIFICATE VERIFICATION", header_style))
            story.append(Paragraph("Scan the QR code below to verify this certificate online:", styles['Normal']))
            story.append(Spacer(1, 12))
            
            qr_image = Image(temp_qr_path, width=1.5*inch, height=1.5*inch)
            qr_image.hAlign = 'CENTER'
            story.append(qr_image)
            story.append(Spacer(1, 12))
            
            story.append(Paragraph(f"Verification URL: {qr_data}", styles['Normal']))
            story.append(Spacer(1, 30))

        finally:
            # Clean up temporary file
            if os.path.exists(temp_qr_path):
                os.unlink(temp_qr_path)

        # Footer
        footer_text = f"""
        This certificate was generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} by the 
        Carbon Credit Verification System. This document certifies that the above project has been 
        verified using AI-powered satellite analysis and meets the required standards for carbon credit generation.
        """
        story.append(Paragraph(footer_text, styles['Normal']))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    def generate_analytics_report(self, analytics_data, user_name="System Administrator"):
        """Generate a comprehensive analytics report"""
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Title
        story.append(Paragraph("CARBON CREDIT VERIFICATION", title_style))
        story.append(Paragraph("ANALYTICS REPORT", title_style))
        story.append(Spacer(1, 30))
        
        # Report metadata
        metadata = [
            ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Generated By:', user_name],
            ['Report Period:', 'All Time'],
            ['System Version:', '1.0.0']
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 30))

        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", styles['Heading2']))
        
        summary_data = analytics_data.get('dashboard', {})
        ml_perf = summary_data.get('ml_performance', {})
        forest_f1 = ml_perf.get('forest_cover_f1', 0)
        change_f1 = ml_perf.get('change_detection_f1', 0)
        summary_text = f"""
        The Carbon Credit Verification System has processed {summary_data.get('total_projects', 0)} projects
        with {summary_data.get('total_verifications', 0)} completed verifications, covering
        {summary_data.get('total_carbon_impact', 0):.2f} tonnes of CO₂ equivalent (applicant-estimated).
        Model performance is reported from offline evaluation: forest-cover F1 {forest_f1:.2f},
        change-detection F1 {change_f1:.2f}.
        """

        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Key Metrics
        story.append(Paragraph("KEY PERFORMANCE INDICATORS", styles['Heading2']))

        kpi_data = [
            ['Metric', 'Value'],
            ['Total Projects', str(summary_data.get('total_projects', 0))],
            ['Verified Projects', str(len([s for s in summary_data.get('project_status', {}).keys() if s == 'Verified']))],
            ['Total Carbon Impact (estimated)', f"{summary_data.get('total_carbon_impact', 0):.2f} tonnes CO₂e"],
            ['Forest-Cover F1 (offline eval)', f"{forest_f1:.2f}"],
            ['Change-Detection F1 (offline eval)', f"{change_f1:.2f}"],
        ]

        kpi_table = Table(kpi_data, colWidths=[3.5*inch, 2.5*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(kpi_table)
        story.append(Spacer(1, 30))

        # Recommendations
        story.append(Paragraph("RECOMMENDATIONS", styles['Heading2']))
        recommendations = [
            "Continue monitoring ML model performance to maintain high accuracy",
            "Implement additional verification steps for high-value carbon credits",
            "Expand geographical coverage to increase project diversity",
            "Develop automated reporting for stakeholders"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer 
# User Roles and Permissions - Carbon Credit Verification System

## 🎭 **Role-Based Access Control (RBAC)**

This document outlines the permissions, capabilities, and access levels for each user role in the Carbon Credit Verification System.

---

## 🛡️ **Administrative Roles**

### **Admin**
**Users:** `testadmin@example.com`, `admin@admin.com`

#### **Permissions:**
- ✅ Full system access
- ✅ User management (create, edit, delete users)
- ✅ View all projects across all users
- ✅ Modify any project data
- ✅ Access all verification records
- ✅ System configuration and settings
- ✅ Database management
- ✅ ML model management
- ✅ XAI access for all projects

#### **Capabilities:**
- Dashboard: Complete system overview
- Projects: CRUD operations on all projects
- Verification: Full verification workflow control
- Analytics: System-wide analytics and reports
- XAI: Access to all explanations and comparisons
- User Management: Add/remove/modify users

---

## 🔍 **Verification & Quality Assurance Roles**

### **Verifier**
**Users:** `verifier@example.com`, `bob@example.com`

#### **Permissions:**
- ✅ View assigned projects for verification
- ✅ Approve/reject carbon credit claims
- ✅ Create verification reports
- ✅ Access ML analysis results
- ✅ Generate XAI explanations for verification decisions
- ❌ Cannot modify project base data
- ❌ Cannot access other verifiers' assignments

#### **Capabilities:**
- Dashboard: Verification queue and pending reviews
- Projects: Read-only access to project details
- Verification: Create, update verification records
- Analytics: Verification-specific metrics
- XAI: Generate explanations for verification decisions
- Reports: Create verification reports and certificates

### **Auditor**
**Users:** `auditor@example.com`

#### **Permissions:**
- ✅ Independent review of verification decisions
- ✅ Access to verification audit trails
- ✅ Generate audit reports
- ✅ Review ML model decisions
- ✅ Access XAI explanations for audit purposes
- ❌ Cannot approve/reject initial verifications
- ❌ Cannot modify project data

#### **Capabilities:**
- Dashboard: Audit queue and compliance overview
- Projects: Read-only access with audit focus
- Verification: Review and audit existing verifications
- Analytics: Audit and compliance metrics
- XAI: Review ML model explanations
- Reports: Generate audit and compliance reports

---

## 🧪 **Scientific & Research Roles**

### **Scientist**
**Users:** `scientist@example.com`

#### **Permissions:**
- ✅ Access ML analysis tools
- ✅ Run advanced environmental analysis
- ✅ Generate detailed XAI explanations
- ✅ Access research datasets
- ✅ Create scientific reports
- ❌ Cannot approve verifications
- ❌ Limited project modification rights

#### **Capabilities:**
- Dashboard: Scientific analysis overview
- Projects: Focus on environmental data and analysis
- Verification: Scientific input and recommendations
- Analytics: Advanced environmental metrics
- XAI: Deep dive into model explanations
- Research: Access to research tools and datasets

### **Researcher**
**Users:** `researcher@example.com`

#### **Permissions:**
- ✅ Access research data and analytics
- ✅ Generate research reports
- ✅ Use XAI for research purposes
- ✅ Access historical data
- ✅ Export data for research
- ❌ Cannot modify verification status
- ❌ Read-only access to live projects

#### **Capabilities:**
- Dashboard: Research data overview
- Projects: Historical and aggregate data access
- Verification: Research analysis of verification patterns
- Analytics: Advanced research analytics
- XAI: Research-focused explanations
- Export: Data export for external research

---

## 💰 **Business & Finance Roles**

### **Investor**
**Users:** `investor@example.com`

#### **Permissions:**
- ✅ View verified carbon credit portfolios
- ✅ Access investment analytics
- ✅ Generate investment reports
- ✅ View market valuations
- ✅ Access XAI for investment decisions
- ❌ Cannot modify verification status
- ❌ Cannot access unverified projects

#### **Capabilities:**
- Dashboard: Investment portfolio overview
- Projects: Focus on verified projects and ROI
- Verification: Investment-grade verification status
- Analytics: Financial and investment metrics
- XAI: Investment decision explanations
- Reports: Investment and portfolio reports

### **Broker**
**Users:** `broker@example.com`

#### **Permissions:**
- ✅ Access carbon credit marketplace
- ✅ Facilitate credit trading
- ✅ Generate trading reports
- ✅ Access market analytics
- ✅ View verification certificates
- ❌ Cannot modify verification data
- ❌ Cannot access internal project details

#### **Capabilities:**
- Dashboard: Market and trading overview
- Projects: Market-ready verified credits
- Verification: Certificate and trading status
- Analytics: Market and trading analytics
- Reports: Trading and market reports
- Marketplace: Credit trading interfaces

---

## 🏛️ **Regulatory & Compliance Roles**

### **Regulator**
**Users:** `regulator@example.com`

#### **Permissions:**
- ✅ Full compliance oversight
- ✅ Access all verification records
- ✅ Generate regulatory reports
- ✅ Audit verification processes
- ✅ Access XAI for regulatory review
- ✅ Set compliance standards
- ❌ Cannot directly modify project data

#### **Capabilities:**
- Dashboard: Regulatory compliance overview
- Projects: Compliance-focused project review
- Verification: Regulatory verification oversight
- Analytics: Compliance and regulatory metrics
- XAI: Regulatory decision explanations
- Reports: Regulatory and compliance reports

### **Monitor**
**Users:** `monitor@example.com`

#### **Permissions:**
- ✅ Continuous monitoring of projects
- ✅ Environmental impact tracking
- ✅ Generate monitoring reports
- ✅ Access real-time data
- ✅ Alert generation for anomalies
- ❌ Cannot approve verifications
- ❌ Cannot modify project parameters

#### **Capabilities:**
- Dashboard: Real-time monitoring overview
- Projects: Environmental monitoring focus
- Verification: Monitoring data for verification
- Analytics: Environmental monitoring metrics
- Alerts: Real-time anomaly detection
- Reports: Environmental monitoring reports

---

## 👷 **Project Development Roles**

### **Project Developer**
**Users:** `test@example.com`, `alice@example.com`, `charlie@example.com`, etc.

#### **Permissions:**
- ✅ Create and manage own projects
- ✅ Submit projects for verification
- ✅ Access own project analytics
- ✅ Generate project reports
- ✅ Use ML analysis tools
- ✅ Access XAI for own projects
- ❌ Cannot access other users' projects
- ❌ Cannot approve own verifications

#### **Capabilities:**
- Dashboard: Personal project overview
- Projects: Full CRUD on own projects
- Verification: Submit for verification, view status
- Analytics: Own project analytics
- XAI: Project-specific explanations
- Reports: Project development reports

---

## 🔐 **Access Matrix Summary**

| Feature | Admin | Verifier | Auditor | Scientist | Researcher | Investor | Broker | Regulator | Monitor | Developer |
|---------|-------|----------|---------|-----------|------------|----------|---------|-----------|---------|-----------|
| **View All Projects** | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | ❌ |
| **Create Projects** | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Verify Projects** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| **Audit Verifications** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **ML Analysis** | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ |
| **XAI Access** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ |
| **User Management** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **System Config** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ |

**Legend:**
- ✅ Full Access
- ⚠️ Limited/Conditional Access  
- ❌ No Access

---

## 🚀 **Testing Recommendations**

### **Admin Testing**
Use `testadmin@example.com` to test:
- Complete system functionality
- User management features
- System administration
- Cross-role data access

### **Workflow Testing**
Use role-specific accounts to test:
- **Verification:** `verifier@example.com` → `auditor@example.com`
- **Research:** `scientist@example.com` → `researcher@example.com`
- **Business:** `investor@example.com` → `broker@example.com`
- **Compliance:** `regulator@example.com` → `monitor@example.com`

### **Security Testing**
- Verify role restrictions are enforced
- Test unauthorized access attempts
- Validate data isolation between roles
- Confirm permission boundaries

---

*Last Updated: December 2024*
*Role-Based Access Control Version: 2.0* 
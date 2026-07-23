# Testing Scenarios - Carbon Credit Verification System

## 🧪 **Comprehensive Testing Guide**

This document provides specific testing scenarios for each user role to ensure system functionality and proper access control.

### 🔑 **Test user password: provisioned via environment/seeding (the admin from `ADMIN_EMAIL` / `ADMIN_PASSWORD`); not hardcoded here**

---

## 🛡️ **Administrative Testing**

### **Scenario A1: System Administration**
**User:** `testadmin@example.com`
**Objective:** Test full system administration capabilities

**Test Steps:**
1. Login as test administrator
2. Access Dashboard → Should see system-wide overview
3. Navigate to Projects → Should see ALL projects (16, 17, 18, etc.)
4. Navigate to Verification → Should access all verification records
5. Test XAI → Should access explanations for any project
6. Check map functionality → Should see focused project areas

**Expected Results:**
- ✅ Complete system access
- ✅ All projects visible with geometry data
- ✅ Maps focus on project locations (Costa Rica, Amazon, Madagascar)
- ✅ All verification data accessible

---

## 🔍 **Verification Workflow Testing**

### **Scenario V1: Carbon Credit Verification**
**User:** `verifier@example.com`
**Objective:** Test verification workflow and decision-making

**Test Steps:**
1. Login as carbon verifier
2. Navigate to Projects → View projects requiring verification
3. Select Project 16 (Costa Rica) → Verify map focuses on Monteverde
4. Access Verification tab → Create new verification record
5. Use XAI → Generate explanation for verification decision
6. Submit verification with confidence score

**Expected Results:**
- ✅ Access to verification-pending projects
- ✅ Map focuses on Costa Rica Cloud Forest
- ✅ Can create verification records
- ✅ XAI explanations available

### **Scenario V2: Independent Audit**
**User:** `auditor@example.com`
**Objective:** Test audit capabilities and independence

**Test Steps:**
1. Login as third-party auditor
2. Review existing verifications from previous scenario
3. Access Project 17 (Amazon) → Verify map focuses on Rondônia
4. Generate audit report for verification decisions
5. Use XAI → Review ML model explanations

**Expected Results:**
- ✅ Access to completed verifications
- ✅ Map focuses on Amazon region
- ✅ Can generate audit reports
- ✅ Cannot modify verification status

---

## 🧪 **Scientific Analysis Testing**

### **Scenario S1: Environmental Analysis**
**User:** `scientist@example.com`
**Objective:** Test scientific analysis and ML capabilities

**Test Steps:**
1. Login as environmental scientist
2. Navigate to Project 18 (Madagascar)
3. Verify map focuses on Andasibe-Mantadia National Park
4. Access ML Analysis → Run forest cover analysis
5. Use XAI → Generate detailed scientific explanations
6. Export analysis results

**Expected Results:**
- ✅ Access to scientific analysis tools
- ✅ Map focuses on Madagascar park
- ✅ ML models operational
- ✅ Detailed XAI explanations available

### **Scenario S2: Research Data Analysis**
**User:** `researcher@example.com`
**Objective:** Test research capabilities and data access

**Test Steps:**
1. Login as climate researcher
2. Access historical project data
3. Navigate between all three projects
4. Verify map geometry works for all locations
5. Generate research analytics
6. Export data for external research

**Expected Results:**
- ✅ Access to historical data
- ✅ All project maps focus correctly
- ✅ Research analytics available
- ✅ Data export functionality

---

## 💰 **Business Operations Testing**

### **Scenario B1: Investment Analysis**
**User:** `investor@example.com`
**Objective:** Test investment perspective and verified credits

**Test Steps:**
1. Login as carbon credit investor
2. View verified projects portfolio
3. Check Project 16 verification status (should be "Verified")
4. Verify carbon impact: 2,156.8 tons CO2/year
5. Use XAI for investment decision explanations
6. Generate investment reports

**Expected Results:**
- ✅ Access to verified projects only
- ✅ Verification data shows correct carbon impact
- ✅ XAI supports investment decisions
- ✅ Investment analytics available

### **Scenario B2: Credit Trading**
**User:** `broker@example.com`
**Objective:** Test trading and marketplace functionality

**Test Steps:**
1. Login as carbon credit broker
2. Access marketplace for verified credits
3. Review Project 17 (Amazon) - 15,750.0 tons CO2/year
4. Check certificate IDs and trading status
5. Generate trading reports

**Expected Results:**
- ✅ Access to market-ready credits
- ✅ Verification certificates visible
- ✅ Trading analytics available
- ✅ Cannot modify verification data

---

## 🏛️ **Regulatory Compliance Testing**

### **Scenario R1: Regulatory Oversight**
**User:** `regulator@example.com`
**Objective:** Test regulatory compliance and oversight

**Test Steps:**
1. Login as environmental regulator
2. Review compliance across all projects
3. Check Project 18 (Madagascar) verification
4. Verify carbon impact: 3,192.4 tons CO2/year
5. Access audit trails and compliance reports
6. Use XAI for regulatory review

**Expected Results:**
- ✅ Complete compliance oversight
- ✅ Access to all verification records
- ✅ Audit trails available
- ✅ Regulatory analytics accessible

### **Scenario R2: Environmental Monitoring**
**User:** `monitor@example.com`
**Objective:** Test monitoring and tracking capabilities

**Test Steps:**
1. Login as environmental monitor
2. Access real-time monitoring dashboard
3. Navigate to all three project locations
4. Verify maps focus correctly on each region
5. Check environmental monitoring metrics
6. Generate monitoring reports

**Expected Results:**
- ✅ Real-time monitoring access
- ✅ All maps focus on correct locations
- ✅ Environmental metrics available
- ✅ Monitoring reports functional

---

## 👷 **Project Development Testing**

### **Scenario P1: Project Creation and Management**
**User:** `test@example.com`
**Objective:** Test project developer capabilities

**Test Steps:**
1. Login as project developer
2. Create new project with geometry data
3. Test map component with new location
4. Submit project for verification
5. Access own project analytics
6. Use XAI for project optimization

**Expected Results:**
- ✅ Can create and manage own projects
- ✅ Map component works with new geometry
- ✅ Can submit for verification
- ✅ Cannot access other users' projects

---

## 🗺️ **Map Functionality Testing**

### **Critical Map Tests:**
**All Users - Test Each Location**

1. **Costa Rica (Project 16)**
   - Map should focus on coordinates: (10.2865, -84.7975)
   - Should display Monteverde Cloud Forest area
   - Polygon boundary should be visible

2. **Amazon Brazil (Project 17)**
   - Map should focus on coordinates: (-8.8250, -63.7750)
   - Should display Rondônia State area
   - Amazon region clearly visible

3. **Madagascar (Project 18)**
   - Map should focus on coordinates: (-18.9000, 48.4500)
   - Should display Andasibe-Mantadia National Park
   - Madagascar island boundaries visible

**Test Result:** ✅ No more global view - all maps focus on project areas

---

## 🔐 **Security and Access Control Testing**

### **Cross-Role Security Tests:**

1. **Unauthorized Access Test**
   - Login as `investor@example.com`
   - Try to access verification approval functions
   - **Expected:** Access denied

2. **Data Isolation Test**
   - Login as `test@example.com`
   - Try to access other users' projects
   - **Expected:** Only own projects visible

3. **Permission Boundary Test**
   - Login as `verifier@example.com`
   - Try to modify project base data
   - **Expected:** Read-only access

---

## 📊 **System Performance Testing**

### **Load Testing Scenarios:**

1. **Multiple User Login**
   - Login with 5 different user types simultaneously
   - Test system responsiveness

2. **ML Model Performance**
   - Run ML analysis as `scientist@example.com`
   - Measure response times for model inference

3. **XAI Generation**
   - Generate explanations as different user types
   - Test explanation generation speed

---

## ✅ **Test Checklist**

### **Core Functionality:**
- [ ] Login works for all user types
- [ ] Maps focus on correct project locations
- [ ] Verification data displays correctly
- [ ] XAI explanations generate successfully
- [ ] Role-based access control enforced

### **Data Integrity:**
- [ ] Project 16: 2,156.8 tons CO2/year, 88.1% confidence
- [ ] Project 17: 15,750.0 tons CO2/year, 92.5% confidence
- [ ] Project 18: 3,192.4 tons CO2/year, 84.6% confidence

### **Geographic Accuracy:**
- [ ] Costa Rica map centers on Monteverde
- [ ] Amazon map centers on Rondônia State
- [ ] Madagascar map centers on Andasibe-Mantadia

### **Security:**
- [ ] Users cannot access unauthorized data
- [ ] Role permissions enforced correctly
- [ ] Authentication working properly

---

## 🚀 **Quick Test Commands**

```bash
# Start backend with virtual environment
source .venv/bin/activate && cd backend && python3 main.py &

# Verify backend is running
curl -s "http://localhost:8000/health"

# Test login (replace with actual user)
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=$ADMIN_EMAIL&password=$ADMIN_PASSWORD"
```

---

*Testing Guide Version: 2.0*
*Last Updated: December 2024*
*Total Test Scenarios: 12* 
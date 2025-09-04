# Security Assistant Frontend

A simple React TypeScript application for the Security Incident Knowledge Assistant.

## Features

- **Role Selection**: Choose between Security Team or Sales Team roles
- **Real-time Chat**: Interactive chat interface with the AI assistant
- **Sample Queries**: Pre-built queries for different user roles
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: User-friendly error messages and loading states

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

3. **Make sure the backend is running**:
   - Backend should be running on `http://localhost:8000`
   - See `../backend/SETUP.md` for backend instructions

4. **Open in browser**: http://localhost:3000

## User Roles

### Security Team
- Full access to all security policies and procedures
- Can view all log entries including critical incidents
- Access to incident escalation procedures

### Sales Team  
- Access to basic security policies
- Limited log access (sales department only)
- No access to sensitive incident escalation procedures

## Sample Queries

### For Security Team:
- "How should I handle a suspected phishing email?"
- "Show me today's failed login attempts from the logs"
- "What's the escalation path for a security breach?"
- "What are the contact details for critical incidents?"

### For Sales Team:
- "How should I handle a suspicious email?"
- "What's our policy for customer data protection?"
- "Show me recent login activity for sales team"
- "What should I do if I suspect a phishing attempt?"

## Technical Details

- **React 19** with TypeScript
- **Latest dependencies** as of 2024/2025
- **No external CSS frameworks** - pure CSS with modern styling
- **Responsive design** with mobile-first approach
- **Error boundaries** and loading states
- **Auto-scroll** to latest messages
- **Typing indicators** for better UX

## API Integration

The frontend communicates with the FastAPI backend:
- **Endpoint**: `POST http://localhost:8000/chat`
- **CORS**: Enabled for `http://localhost:3000`
- **Error Handling**: Displays user-friendly error messages

## Build for Production

```bash
npm run build
```

The build folder will contain the optimized production build.

## Project Structure

```
src/
├── App.tsx          # Main application component
├── App.css          # All styles in one file
├── index.tsx        # React entry point
└── ...

public/
├── index.html       # HTML template
└── ...
```

## Browser Support

- Chrome (latest)
- Firefox (latest) 
- Safari (latest)
- Edge (latest)
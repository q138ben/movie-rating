# Movie Rating Predictor Frontend

The frontend of the Movie Rating Predictor application, built with Next.js 14, TypeScript, and Tailwind CSS.

## Features

- Modern, responsive UI
- Dynamic form with real-time validation
- Support for multiple actors, directors, and studios
- Loading states and error handling
- Beautiful results display with confidence indicators

## Tech Stack

- Next.js 14
- TypeScript
- Tailwind CSS
- React Hook Form (for form handling)
- Axios (for API calls)

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── page.tsx           # Main page component
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   └── components/            # React components
│       ├── MovieForm.tsx      # Movie input form
│       └── Results.tsx        # Results display
├── public/                    # Static assets
└── package.json              # Dependencies and scripts
```

## Components

### MovieForm

A dynamic form component that handles:
- Basic movie information (title, description, tagline)
- Numerical inputs (release year, duration)
- Categorical selection (age rating)
- Dynamic arrays (actors, directors, studios)

### Results

Displays the prediction results with:
- Predicted rating (0-10 scale)
- Confidence level indicator
- Visual feedback through color coding

## API Integration

The frontend communicates with the backend API at `http://localhost:8000`:

```typescript
// Example API call
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(movieData),
});
```

## Development

### Adding New Features

1. Create new components in `src/components/`
2. Update types in `src/types/` if needed
3. Add new routes in `src/app/` if required
4. Update the main page to include new features

### Styling

The application uses Tailwind CSS for styling. To add new styles:

1. Use Tailwind utility classes directly in components
2. Add custom styles in `src/app/globals.css`
3. Create new components for reusable styled elements

### Testing

Run tests with:
```bash
npm test
```

## Deployment

1. Build the application:
```bash
npm run build
```

2. Start the production server:
```bash
npm start
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

This project is licensed under the MIT License.

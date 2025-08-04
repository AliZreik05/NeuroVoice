import React from 'react';

// A simple Button component to mimic the shadcn/ui Button.
const Button = ({ children, className = '', variant = 'default', ...props }) => {
  const baseStyles = 'px-6 py-3 font-semibold rounded-full transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  let variantStyles = '';
  switch (variant) {
    case 'outline':
      variantStyles = 'border border-current bg-transparent text-white hover:bg-white hover:text-blue-800';
      break;
    case 'default':
    default:
      variantStyles = 'bg-white text-blue-800 hover:bg-gray-100';
      break;
  }
  
  const combinedClassName = `${baseStyles} ${variantStyles} ${className}`;
  
  return (
    <button className={combinedClassName} {...props}>
      {children}
    </button>
  );
};

// The main LandingPage component.
function LandingPage() {
  return (
    <div className="bg-white text-gray-900 font-sans">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-900 to-purple-800 text-white py-20 px-6 text-center rounded-b-[4rem] md:rounded-b-[6rem]">
        <h1 className="text-4xl md:text-6xl font-bold mb-4 animate-fade-in-down">
          Detect Alzheimer's from Speech
        </h1>
        <p className="text-lg md:text-xl mb-8 max-w-2xl mx-auto animate-fade-in">
          Our LLM-powered AI model analyzes your speech to detect early signs of Alzheimer’s — fast, accurate, and non-invasive.
        </p>
        <div className="space-x-4 flex justify-center flex-wrap gap-4 animate-fade-in-up">
          <Button className="bg-white text-blue-800 hover:bg-gray-100 shadow-lg">Try the Demo</Button>
          <Button variant="outline" className="border-white text-white hover:bg-white hover:text-blue-800 shadow-lg">
            Learn More
          </Button>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 px-6 text-center max-w-5xl mx-auto">
        <h2 className="text-3xl font-semibold mb-10">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="p-8 shadow-xl rounded-2xl bg-gray-50 hover:scale-105 transition-transform duration-300">
            <h3 className="text-2xl font-bold mb-4">1. Record Your Voice</h3>
            <p className="text-gray-600">Use your device to record a short voice sample securely.</p>
          </div>
          <div className="p-8 shadow-xl rounded-2xl bg-gray-50 hover:scale-105 transition-transform duration-300">
            <h3 className="text-2xl font-bold mb-4">2. Upload & Analyze</h3>
            <p className="text-gray-600">Submit your voice and let our LLM analyze patterns linked to cognitive decline.</p>
          </div>
          <div className="p-8 shadow-xl rounded-2xl bg-gray-50 hover:scale-105 transition-transform duration-300">
            <h3 className="text-2xl font-bold mb-4">3. Get Results</h3>
            <p className="text-gray-600">Receive instant, research-backed feedback and recommendations.</p>
          </div>
        </div>
      </section>

      {/* Why It Matters Section */}
      <section className="bg-gray-100 py-20 px-6 text-center">
        <h2 className="text-3xl font-semibold mb-6">Why Early Detection Matters</h2>
        <p className="max-w-3xl mx-auto mb-6 text-lg text-gray-700">
          Alzheimer's is most treatable in its earliest stages. Our technology empowers individuals and caregivers by offering a fast and accessible screening tool — anytime, anywhere.
        </p>
        <Button className="bg-blue-800 text-white hover:bg-blue-700 shadow-lg">Join the Mission</Button>
      </section>

      {/* Accuracy Section */}
      <section className="py-20 px-6 text-center max-w-4xl mx-auto">
        <h2 className="text-3xl font-semibold mb-6">Backed by Research</h2>
        <p className="mb-4 text-lg text-gray-700">
          Our model has been validated on clinical datasets with over 90% accuracy in detecting early signs of Alzheimer’s.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          <div className="bg-white p-8 shadow-xl rounded-2xl hover:scale-105 transition-transform duration-300">
            <h4 className="text-xl font-semibold mb-2">90%+ Accuracy</h4>
            <p className="text-gray-600">Achieved using voice samples tested against benchmark datasets.</p>
          </div>
          <div className="bg-white p-8 shadow-xl rounded-2xl hover:scale-105 transition-transform duration-300">
            <h4 className="text-xl font-semibold mb-2">LLM-Based Model</h4>
            <p className="text-gray-600">Utilizes the latest advancements in language and speech modeling.</p>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="bg-blue-900 text-white py-16 text-center px-6 rounded-t-[4rem] md:rounded-t-[6rem]">
        <h2 className="text-3xl font-bold mb-4">Be Part of the Change</h2>
        <p className="mb-6 max-w-2xl mx-auto text-lg">
          Help us improve early Alzheimer’s detection. Try the tool, give feedback, or join our research network.
        </p>
        <Button className="bg-white text-blue-900 hover:bg-gray-100 shadow-lg">Get Started</Button>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-300 py-10 px-6 text-center">
        <p className="mb-4">© 2025 Alzheimer Voice AI. All rights reserved.</p>
        <div className="space-x-4">
          <a href="#" className="hover:underline">Privacy Policy</a>
          <a href="#" className="hover:underline">Terms of Service</a>
          <a href="#" className="hover:underline">GitHub</a>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;

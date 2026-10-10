import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  afterEach(() => {
    try {
      TestBed.inject(HttpTestingController).verify();
    } finally {
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
      localStorage.clear();
      TestBed.resetTestingModule();
    }
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    TestBed.inject(HttpTestingController)
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the dashboard and online API state', async () => {
    const fixture = TestBed.createComponent(App);
    const request = TestBed.inject(HttpTestingController).expectOne('/api/health');
    request.flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Overview');
    expect(compiled.querySelector('.api-pill')?.textContent).toContain('API connected');
    expect(compiled.textContent).toContain('Assessment workflow');
  });

  it('should poll live generation progress and send the selected validation mode', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const app = fixture.componentInstance;
    app.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.com',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    app.query.set('binary search trees');
    expect(app.validationMode()).toBe('fast');
    const generation = app.generate();
    const request = http.expectOne('/api/questions/generate/start');

    expect(request.request.body.validation_mode).toBe('fast');
    request.flush({
      id: 'run-live',
      status: 'running',
      request_type: 'single',
      stage: 'retrieval',
      message: 'Retrieving evidence.',
      started_at: new Date().toISOString(),
      logs: [{
        timestamp: new Date().toISOString(),
        stage: 'model_route',
        level: 'info',
        message: 'Successful model route: OpenRouter route 1 (openrouter, model=example).',
      }],
    });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(app.liveGenerationRun()?.stage).toBe('retrieval');
    expect(app.liveGenerationRoute()).toContain('OpenRouter route 1');
    await new Promise((resolve) => setTimeout(resolve, 720));
    http.expectOne('/api/questions/generate/runs/run-live').flush({
      id: 'run-live',
      status: 'completed',
      request_type: 'single',
      stage: 'completed',
      message: 'Generation completed successfully.',
      started_at: new Date().toISOString(),
      result: [{
        question_text: 'Which subtree stores smaller values?',
        difficulty: 'Medium',
        bloom_level: 'Apply',
        options: [{ key: 'A', text: 'Left' }],
        correct_answer: 'A',
        explanation: 'Smaller values are stored on the left.',
      }],
      logs: [],
      metrics: { total_duration_ms: 100 },
    });
    await new Promise((resolve) => setTimeout(resolve, 10));
    http.expectOne('/api/evaluation/questions').flush({
      total_questions: 1,
      unique_questions: 1,
      duplicate_count: 0,
      duplicate_rate: 0,
      coverage: 1,
      difficulty_distribution: { Medium: 1 },
      bloom_distribution: { Apply: 1 },
      pattern_distribution: {},
    });
    await generation;
    http.expectOne('/api/questions/usage').flush({
      daily_limit: 50,
      used_today: 1,
      remaining_today: 49,
      questions_created: 1,
    });
  });

  it('should show the practice setup and report an unauthenticated start', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();

    const buttons = Array.from(
      fixture.nativeElement.querySelectorAll('button'),
    ) as HTMLButtonElement[];
    const practiceLink = buttons.find((button) =>
      button.textContent?.includes('Practice'),
    ) as HTMLButtonElement;
    practiceLink.click();
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Practice with purpose.');
    (fixture.nativeElement.querySelector('.start-button') as HTMLButtonElement).click();
    const startRequest = http.expectOne('/api/practice/start');
    expect(startRequest.request.body).toEqual({
      subject_id: 'computer-science',
      question_count: 10,
      difficulty: null,
    });
    startRequest.flush(
      { detail: 'Authentication required' },
      { status: 401, statusText: 'Unauthorized' },
    );
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.practice-message')?.textContent).toContain(
      'Sign in as a student',
    );
  });

  it('should upload material without a subject and generate from the selected pattern', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    const component = fixture.componentInstance as unknown as {
      currentUser: {
        set(user: { id: string; email: string; display_name: string; roles: string[] }): void;
      };
    };
    component.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.com',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const questionPaperLink = (
      Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[]
    ).find((button) => button.textContent?.includes('Generate questions')) as HTMLButtonElement;
    questionPaperLink.click();
    const templatesRequest = http.expectOne('/api/templates');
    templatesRequest.flush([
      {
        _id: 'pattern-1',
        name: 'GIS paper',
        question_type: 'MCQ',
        pattern: 'Direct Concept',
        supported_difficulties: ['Easy', 'Medium', 'Hard'],
        supported_bloom_levels: ['Understand'],
        marks: 1,
        total_marks: 1,
        sections: [{ question_type: 'MCQ', pattern: 'Direct Concept', count: 1, marks: 1 }],
      },
    ]);
    await fixture.whenStable();
    fixture.detectChanges();

    const fileInput = fixture.nativeElement.querySelector('.file-field input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      value: [new File(['study material'], 'Module 2.pdf', { type: 'application/pdf' })],
    });
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((resolve) => setTimeout(resolve, 50));
    fixture.detectChanges();

    const generateButton = fixture.nativeElement.querySelector(
      '.start-button',
    ) as HTMLButtonElement;
    expect(generateButton.disabled).toBe(false);
    generateButton.click();
    const uploadRequest = http.expectOne('/api/materials/upload');
    expect(uploadRequest.request.body.get('subject_id')).toBeNull();
    uploadRequest.flush({
      _id: 'material-1',
      filename: 'Module 2.pdf',
      status: 'uploaded',
      size_bytes: 13,
    });
    await new Promise((resolve) => setTimeout(resolve, 300));
    const processRequest = http.expectOne('/api/materials/material-1/process');
    processRequest.flush({ chunk_count: 1 });
    await fixture.whenStable();
    const indexRequest = http.expectOne('/api/retrieval/materials/material-1/index');
    indexRequest.flush({ indexed_chunks: 1, embedding_model: 'hashing-v1' });
    await fixture.whenStable();
    const generateRequest = http.expectOne('/api/questions/generate/paper/start');
    expect(generateRequest.request.body).toEqual({
      template_id: 'pattern-1',
      material_id: 'material-1',
      top_k: 5,
      validation_mode: 'strict',
    });
    generateRequest.flush({
      id: 'run-1', status: 'completed', result: [{
        id: 'question-1',
        question_text: 'What is a geographic information system?',
        question_type: 'MCQ',
        pattern: 'Direct Concept',
        marks: 1,
        difficulty: 'Easy',
        bloom_level: 'Understand',
        options: [
          { key: 'A', text: 'A mapping system' },
          { key: 'B', text: 'A database only' },
        ],
        explanation: 'A GIS captures, stores, analyzes, and presents geographic data.',
        sources: [{ page_number: 1, text: 'A geographic information system...' }],
      }],
    });
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    http.expectOne('/api/evaluation/questions').flush({
      total_questions: 1,
      unique_questions: 1,
      duplicate_count: 0,
      duplicate_rate: 0,
      coverage: 1,
      difficulty_distribution: { Easy: 1 },
      bloom_distribution: { Understand: 1 },
      pattern_distribution: { 'Direct Concept': 1 },
    });
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();

    expect(
      (fixture.componentInstance as unknown as { view: () => string }).view(),
    ).toBe('Generated questions');
    expect(fixture.nativeElement.textContent).toContain('What is a geographic information system?');
  });

  it('should create and select a real subject instead of submitting a placeholder ID', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    const app = fixture.componentInstance;
    app.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.com',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    expect(app.subjects()).toEqual([]);
    expect(app.selectedSubjectId()).toBe('');
    app.newSubjectCode.set('CS301');
    app.newSubjectName.set('Algorithms');

    const creation = app.createSubject();
    const request = http.expectOne('/api/subjects');
    expect(request.request.body).toEqual({ code: 'CS301', name: 'Algorithms' });
    request.flush({ _id: 'subject-301', code: 'CS301', name: 'Algorithms' });
    await creation;

    expect(app.subjects()).toEqual([
      { _id: 'subject-301', id: 'subject-301', code: 'CS301', name: 'Algorithms' },
    ]);
    expect(app.selectedSubjectId()).toBe('subject-301');
  });

  it('should stop paper generation until a file is selected', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    const app = fixture.componentInstance;
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    await app.uploadAndGeneratePaper();

    http.expectNone('/api/materials/upload');
    expect(app.notice()).toContain('Upload the study material');
  });

  it('should save a mixed question-pattern blueprint with the matching total marks', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    const app = fixture.componentInstance;
    app.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.com',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    app.newPatternName.set('Mixed midterm paper');
    app.updatePatternSectionNumber(0, 'count', 8);
    app.addPatternSection();
    app.newPatternTotalMarks.set(10);

    const save = app.savePattern();
    const request = http.expectOne('/api/templates');
    expect(request.request.body).toEqual({
      name: 'Mixed midterm paper',
      question_type: 'MCQ',
      pattern: 'Direct Concept & Application',
      required_fields: ['question_text', 'explanation'],
      supported_difficulties: ['Medium'],
      supported_bloom_levels: ['Understand'],
      version: '1.0',
      marks: 1,
      total_marks: 10,
      sections: [
        {
          question_type: 'MCQ',
          pattern: 'Direct Concept & Application',
          count: 8,
          marks: 1,
          supported_difficulties: ['Medium'],
          supported_bloom_levels: ['Understand'],
        },
        {
          question_type: 'Short Answer',
          pattern: 'Direct Concept & Application',
          count: 1,
          marks: 2,
          supported_difficulties: ['Medium'],
          supported_bloom_levels: ['Understand'],
        },
      ],
    });
    request.flush({ id: 'mixed-midterm' });
    await save;

    expect(app.templates()[0]).toMatchObject({
      id: 'mixed-midterm',
      total_marks: 10,
      sections: [
        { question_type: 'MCQ', count: 8, marks: 1 },
        { question_type: 'Short Answer', count: 1, marks: 2 },
      ],
    });
    expect(app.isMixedPattern(app.templates()[0])).toBe(true);
  });

  it('should not save a pattern when its section allocation differs from total marks', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    const app = fixture.componentInstance;
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const originalTemplates = app.templates().length;
    app.newPatternName.set('Mismatched paper');
    app.updatePatternSectionNumber(0, 'count', 8);
    app.addPatternSection();
    app.newPatternTotalMarks.set(11);
    await app.savePattern();

    http.expectNone('/api/templates');
    expect(app.templates()).toHaveLength(originalTemplates);
    expect(app.notice()).toContain('Section marks add up to 10');
  });

  it('should show the sign-in dialog and report invalid credentials', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();

    (fixture.nativeElement.querySelector('.profile') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('#login-title')?.textContent).toContain(
      'Welcome back.',
    );

    (fixture.nativeElement.querySelector('.login-dialog form') as HTMLFormElement).requestSubmit();
    const loginRequest = http.expectOne('/api/auth/login');
    expect(loginRequest.request.body).toEqual({ email: '', password: '' });
    loginRequest.flush(
      { detail: 'Invalid email or password.' },
      { status: 401, statusText: 'Unauthorized' },
    );
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.login-dialog')?.textContent).toContain(
      'Unable to sign in',
    );
  });

  it('should register a faculty account and establish a session', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();

    (fixture.nativeElement.querySelector('.profile') as HTMLButtonElement).click();
    fixture.detectChanges();
    const registerToggle = Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>('.login-dialog button'),
    ).find((button) => button.textContent?.includes('Create an account')) as HTMLButtonElement;
    registerToggle.click();
    fixture.detectChanges();

    const app = fixture.componentInstance;
    app.loginEmail.set('new.faculty@example.edu');
    app.loginPassword.set('secure-passphrase');
    app.registerRole.set('faculty');
    fixture.detectChanges();
    (fixture.nativeElement.querySelector('.login-dialog form') as HTMLFormElement).requestSubmit();

    const registerRequest = http.expectOne('/api/auth/register');
    expect(registerRequest.request.body).toEqual({
      email: 'new.faculty@example.edu',
      password: 'secure-passphrase',
      role: 'faculty',
    });
    registerRequest.flush({ access_token: 'registered-token' });
    await Promise.resolve();

    http.expectOne('/api/auth/me').flush({
      id: 'faculty-1',
      email: 'new.faculty@example.edu',
      display_name: 'new.faculty',
      roles: ['faculty'],
    });
    await Promise.resolve();
    http.expectOne('/api/questions/usage').flush({
      daily_limit: 50,
      used_today: 0,
      remaining_today: 50,
      questions_created: 0,
    });
    http.expectOne('/api/subjects').flush([]);
    http.expectOne('/api/templates').flush([]);
    await fixture.whenStable();
    fixture.detectChanges();

    expect(app.currentUser()?.roles).toEqual(['faculty']);
    expect(app.notice()).toContain('account is ready');
    expect(fixture.nativeElement.textContent).not.toContain('Already registered?');
  });

  it('should populate the library from recently loaded generation sessions', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const app = fixture.componentInstance;
    app.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.edu',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    const loadSessions = app.loadGenerationRuns();
    http.expectOne('/api/questions/generate/runs?limit=100').flush([
      {
        id: 'paper-1',
        request_type: 'paper',
        status: 'completed',
        result: [{ question_text: 'Previously generated question' }],
      },
      {
        id: 'single-1',
        request_type: 'single',
        status: 'completed',
        result: [{ question_text: 'Single question' }],
      },
    ]);
    await loadSessions;

    await app.loadLibraryPapers();

    http.expectNone('/api/questions/generate/runs?limit=100');
    expect(app.libraryPapers().map((paper) => paper.id)).toEqual(['paper-1']);
    expect(app.libraryError()).toBe('');
  });

  it('should display per-stage generation timings and call counts', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const app = fixture.componentInstance;
    app.selectedGenerationRun.set({
      id: 'run-1',
      status: 'completed',
      request_type: 'single',
      metrics: {
        total_duration_ms: 2200,
        retrieval_duration_ms: 25,
        model_duration_ms: 2000,
        generation_duration_ms: 1100,
        critic_duration_ms: 1400,
        solver_duration_ms: 600,
        model_calls: 3,
        generation_calls: 1,
        critic_calls: 1,
        solver_calls: 1,
        candidate_retries: 1,
        jev_decisions: 2,
        jev_review_escalations: 1,
        jev_evidence_confidence_total: 1.42,
      },
      logs: [{
        timestamp: '2026-10-08T12:00:00Z',
        stage: 'generation_model',
        level: 'info',
        message: 'Generation model call took 1.10s.',
        duration_ms: 1100,
      }],
    });
    app.activeView.set('Monitoring');
    fixture.detectChanges();

    const text = (fixture.nativeElement as HTMLElement).textContent || '';
    expect(text).toContain('Total 2.2 s');
    expect(text).toContain('Retrieval 25 ms');
    expect(text).toContain('3 total');
    expect(text).toContain('1.1 s');
    expect(text).toContain('Jev decisions / escalations 2 / 1');
  });

  it('should run a scoped retrieval benchmark and display ranking metrics', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const app = fixture.componentInstance;
    app.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.edu',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    app.navigate('Retrieval benchmark');
    app.updateRetrievalBenchmarkScope('subject_id', 'subject-1');
    app.updateRetrievalBenchmarkCase(0, 'query', '"binary tree traversal"');
    app.updateRetrievalBenchmarkCase(0, 'relevantChunkIds', 'chunk-1, chunk-2');
    app.retrievalBenchmarkK.set(2);

    const benchmarkPromise = app.runRetrievalBenchmark();
    const request = http.expectOne('/api/retrieval/evaluate');
    expect(request.request.body).toEqual({
      cases: [{
        query: '"binary tree traversal"',
        relevant_chunk_ids: ['chunk-1', 'chunk-2'],
      }],
      k: 2,
      subject_id: 'subject-1',
    });
    request.flush({
      strategy: 'hybrid_bm25_keyword_mmr',
      scope: { subject_id: 'subject-1' },
      query_count: 1,
      k: 2,
      metrics: {
        'precision@2': 0.5,
        'recall@2': 0.5,
        'mrr@2': 1,
        'ndcg@2': 0.613,
      },
      per_query: [{
        query: '"binary tree traversal"',
        retrieved_chunk_ids: ['chunk-1', 'unrelated'],
        relevant_chunk_ids: ['chunk-1', 'chunk-2'],
        'precision@2': 0.5,
        'recall@2': 0.5,
        'mrr@2': 1,
        'ndcg@2': 0.613,
      }],
    });
    await benchmarkPromise;
    fixture.detectChanges();

    const text = (fixture.nativeElement as HTMLElement).textContent || '';
    expect(text).toContain('Retrieval Benchmark');
    expect(text).toContain('hybrid_bm25_keyword_mmr');
    expect(text).toContain('Recall@2');
    expect(text).toContain('0.500');
    expect(text).toContain('unrelated');
  });

  it('should hide retrieval benchmark navigation from students', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    const app = fixture.componentInstance;
    app.currentUser.set({
      id: 'student-1',
      email: 'student@example.edu',
      display_name: 'Student',
      roles: ['student'],
    });
    fixture.detectChanges();

    const navLabels = Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll('.nav-links button'),
    ).map((button) => button.textContent || '');
    expect(navLabels.join(' ')).not.toContain('Retrieval eval');
    app.navigate('Retrieval benchmark');
    expect(app.view()).toBe('Dashboard');
  });

  it('should import and export a versioned retrieval benchmark dataset', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    const app = fixture.componentInstance;
    const dataset = {
      version: 1,
      k: 3,
      scope: { subject_id: 'subject-1' },
      cases: [{
        query: 'binary tree traversal',
        relevant_chunk_ids: ['chunk-1', 'chunk-2', 'chunk-1'],
      }],
    };
    app.retrievalBenchmarkJson.set(JSON.stringify(dataset));
    app.importRetrievalBenchmarkJson();

    expect(app.retrievalBenchmarkK()).toBe(3);
    expect(app.retrievalBenchmarkScope().subject_id).toBe('subject-1');
    expect(app.retrievalBenchmarkCases()).toEqual([{
      query: 'binary tree traversal',
      relevantChunkIds: 'chunk-1, chunk-2',
    }]);
    expect(app.retrievalBenchmarkError()).toBe('');

    app.exportRetrievalBenchmarkJson();
    expect(JSON.parse(app.retrievalBenchmarkJson())).toEqual({
      version: 1,
      k: 3,
      scope: { subject_id: 'subject-1' },
      cases: [{
        query: 'binary tree traversal',
        relevant_chunk_ids: ['chunk-1', 'chunk-2'],
      }],
    });
  });

  it('should not change benchmark fields when imported JSON is invalid', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    const app = fixture.componentInstance;
    app.updateRetrievalBenchmarkCase(0, 'query', 'keep this query');
    app.retrievalBenchmarkJson.set(JSON.stringify({
      version: 1,
      k: 30,
      scope: { subject_id: 'subject-1' },
      cases: [{ query: 'replacement', relevant_chunk_ids: ['chunk-1'] }],
    }));

    app.importRetrievalBenchmarkJson();

    expect(app.retrievalBenchmarkCases()[0].query).toBe('keep this query');
    expect(app.retrievalBenchmarkError()).toContain('between 1 and 20');
  });

  it('should download the validated current benchmark as a JSON file', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    const app = fixture.componentInstance;
    app.updateRetrievalBenchmarkScope('subject_id', 'subject-1');
    app.updateRetrievalBenchmarkCase(0, 'query', 'binary tree traversal');
    app.updateRetrievalBenchmarkCase(0, 'relevantChunkIds', 'chunk-1');

    const createObjectURL = vi.fn((_blob: Blob | MediaSource) => 'blob:retrieval-benchmark');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });
    let downloadedName = '';
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloadedName = this.download;
    });

    app.downloadRetrievalBenchmarkJson();

    expect(createObjectURL).toHaveBeenCalledWith(expect.objectContaining({ type: 'application/json' }));
    expect(downloadedName).toBe('retrieval-benchmark.json');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:retrieval-benchmark');
    expect(JSON.parse(app.retrievalBenchmarkJson()).cases[0].query).toBe('binary tree traversal');
  });
  it('tracks actual question completion and the six pipeline stages', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    TestBed.inject(HttpTestingController).expectOne('/api/health').flush({ status: 'ok', service: 'api' });
    const timestamp = '2026-10-09T10:00:00Z';
    app.generationElapsedMs.set(5000);
    app.liveGenerationRun.set({
      id: 'progress-run', status: 'running', started_at: timestamp,
      logs: [
        { timestamp, stage: 'blueprint', level: 'info', message: 'Ready', details: { question_total: 2 } },
        { timestamp, stage: 'preparing', level: 'info', message: 'Preparing', details: { question_index: 1 } },
        { timestamp, stage: 'question_accepted', level: 'info', message: 'Passed', details: { question_index: 1 } },
        { timestamp, stage: 'model_route', level: 'info', message: 'Route', details: { question_index: 1 } },
        { timestamp, stage: 'preparing', level: 'info', message: 'Preparing', details: { question_index: 2 } },
        { timestamp, stage: 'candidate_retry', level: 'info', message: 'Retry', details: { question_index: 2 } },
      ],
    });
    expect(app.validatedQuestionCount()).toBe(1);
    expect(app.questionProgressPercent()).toBe(50);
    expect(app.questionProgress()[0].state).toBe('Validated');
    expect(app.questionProgress()[1].retries).toBe(1);
    expect(app.questionProgress()[1].elapsedMs).toBe(5000);
    expect(app.pipelineSteps('competitive')).toHaveLength(6);
    expect(app.pipelineSteps('competitive')[3].state).toBe('active');
    expect(app.pipelineSteps('competitive')[5].state).toBe('pending');
    app.liveGenerationRun.update((run) => run ? { ...run, status: 'completed' } : null);
    expect(app.questionProgressPercent()).toBe(100);
    expect(app.pipelineSteps('competitive').every((step) => step.state === 'complete')).toBe(true);
  });

});
